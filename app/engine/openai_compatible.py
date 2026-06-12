from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
import time
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import DecodingParams
from app.schemas import EngineResult
from app.schemas import ResponseMetrics
from app.schemas import ResponseRequest

from .common import BackendExecutionError
from .common import LOGGER
from .common import ResolvedDecoding
from .common import _exception_message
from .common import _resolve_request_remote_thinking


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Return only the response."
_REMOTE_API_KIND = "chat_completions"
_REMOTE_HEALTH_CHECK = "config_only"
_REMOTE_THINKING_VALUES = {"enabled", "disabled"}


@dataclass(frozen=True)
class OpenAICompatibleModelRuntime:
    config: ModelSettings
    base_url: str
    api_key_env: str
    remote_model: str
    timeout_s: float


class OpenAICompatibleEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, OpenAICompatibleModelRuntime] = {}
        self._load_errors: dict[str, str] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_settings)
            except Exception as exc:
                self._load_errors[model_name] = _exception_message(exc)
                LOGGER.exception(
                    "Failed to activate remote model '%s' for upstream model '%s'; skipping model.",
                    model_name,
                    model_settings.remote_model,
                )
        if not self._models:
            if self._load_errors:
                details = "; ".join(
                    f"{model_name}: {message}"
                    for model_name, message in sorted(self._load_errors.items())
                )
                raise ValueError(details)
            raise ValueError("no enabled remote models could be loaded")

    def complete(self, request: ResponseRequest) -> EngineResult:
        runtime = self._models.get(request.model)
        if runtime is None:
            raise ValueError(f"unknown model: {request.model!r}")

        decoding = self._resolve_decoding(request.decoding)
        self._log_unsupported_decoding(request)
        payload = self._chat_completions_payload(
            runtime=runtime,
            request=request,
            decoding=decoding,
        )

        started = time.perf_counter()
        response_payload = self._post_json(runtime, payload)
        wall_s = max(0.0, time.perf_counter() - started)

        text = self._extract_text(response_payload)
        prompt_tokens, output_tokens = self._extract_usage(response_payload)
        tokens_per_second = None
        if output_tokens is not None and wall_s > 0.0:
            tokens_per_second = output_tokens / wall_s
        return EngineResult(
            text=text,
            metrics=ResponseMetrics(
                engine_prompt_tokens=prompt_tokens,
                engine_output_tokens=output_tokens,
                engine_tokens_per_second=tokens_per_second,
            ),
        )

    def _build_runtime(self, settings: ModelSettings) -> OpenAICompatibleModelRuntime:
        api_kind = self._required_remote_field(settings.remote_api_kind, "remote_api_kind").lower()
        if api_kind != _REMOTE_API_KIND:
            raise ValueError(f"remote_api_kind must be {_REMOTE_API_KIND!r}")

        health_check = settings.remote_health_check.strip().lower()
        if health_check != _REMOTE_HEALTH_CHECK:
            raise ValueError(f"remote_health_check must be {_REMOTE_HEALTH_CHECK!r} for V1")
        if settings.remote_max_retries != 0:
            raise ValueError("remote_max_retries must be 0 for V1")
        if settings.remote_timeout_s <= 0.0:
            raise ValueError("remote_timeout_s must be greater than 0")
        if settings.remote_thinking is not None and settings.remote_thinking not in _REMOTE_THINKING_VALUES:
            raise ValueError("remote_thinking must be 'enabled', 'disabled', or omitted")

        base_url = self._required_remote_field(settings.remote_base_url, "remote_base_url").rstrip("/")
        api_key_env = self._required_remote_field(settings.remote_api_key_env, "remote_api_key_env")
        remote_model = self._required_remote_field(settings.remote_model, "remote_model")
        if os.environ.get(api_key_env, "").strip() == "":
            raise RuntimeError(f"missing API key environment variable: {api_key_env}")

        return OpenAICompatibleModelRuntime(
            config=settings,
            base_url=base_url,
            api_key_env=api_key_env,
            remote_model=remote_model,
            timeout_s=settings.remote_timeout_s,
        )

    def _chat_completions_payload(
        self,
        *,
        runtime: OpenAICompatibleModelRuntime,
        request: ResponseRequest,
        decoding: ResolvedDecoding,
    ) -> dict[str, object]:
        system_prompt = request.instructions or _DEFAULT_SYSTEM_PROMPT
        user_content = self._user_message_content(request)
        payload: dict[str, object] = {
            "model": runtime.remote_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "max_tokens": decoding.max_tokens,
        }
        if decoding.stop:
            payload["stop"] = decoding.stop
        remote_thinking = _resolve_request_remote_thinking(
            request,
            runtime.config.remote_thinking,
        )
        if remote_thinking is not None:
            payload["thinking"] = {"type": remote_thinking}
        return payload

    @staticmethod
    def _user_message_content(request: ResponseRequest) -> str | list[dict[str, object]]:
        if isinstance(request.input, str):
            return request.input
        return [item.model_dump(mode="python") for item in request.input]

    def _post_json(
        self,
        runtime: OpenAICompatibleModelRuntime,
        payload: dict[str, object],
    ) -> dict[str, object]:
        api_key = os.environ.get(runtime.api_key_env, "").strip()
        if api_key == "":
            raise BackendExecutionError(
                code="missing_api_key_env",
                status_code=500,
                message=f"missing API key environment variable: {runtime.api_key_env}",
            )

        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = Request(
            f"{runtime.base_url}/chat/completions",
            data=data,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=runtime.timeout_s) as response:
                raw_payload = response.read()
        except HTTPError as exc:
            raise self._map_http_error(exc) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise BackendExecutionError(
                code="upstream_timeout",
                status_code=504,
                message="upstream chat completion timed out",
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BackendExecutionError(
                    code="upstream_timeout",
                    status_code=504,
                    message="upstream chat completion timed out",
                ) from exc
            raise BackendExecutionError(
                code="upstream_connection_error",
                status_code=502,
                message="upstream chat completion connection failed",
            ) from exc

        try:
            parsed = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream chat completion returned invalid JSON",
            ) from exc
        if not isinstance(parsed, dict):
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream chat completion returned a non-object JSON response",
            )
        return parsed

    def _map_http_error(self, exc: HTTPError) -> BackendExecutionError:
        status = int(exc.code)
        if status in {401, 403}:
            return BackendExecutionError(
                code="upstream_authentication_failure",
                status_code=502,
                message=f"upstream chat completion authentication failed with HTTP {status}",
            )
        if status == 429:
            return BackendExecutionError(
                code="upstream_rate_limit",
                status_code=429,
                message="upstream chat completion was rate limited",
            )
        if 400 <= status < 500:
            return BackendExecutionError(
                code="upstream_invalid_request",
                status_code=502,
                message=f"upstream chat completion rejected the request with HTTP {status}",
            )
        if status >= 500:
            return BackendExecutionError(
                code="upstream_server_error",
                status_code=502,
                message=f"upstream chat completion failed with HTTP {status}",
            )
        return BackendExecutionError(
            code="upstream_http_error",
            status_code=502,
            message=f"upstream chat completion failed with HTTP {status}",
        )

    def _extract_text(self, payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream chat completion response did not contain choices",
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream chat completion choice was not an object",
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream chat completion choice did not contain a message",
            )
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        raise BackendExecutionError(
            code="upstream_response_parse_failure",
            status_code=502,
            message="upstream chat completion message content was not text",
        )

    def _extract_usage(self, payload: dict[str, object]) -> tuple[int | None, int | None]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None, None
        prompt_tokens = self._coerce_int(usage.get("prompt_tokens"))
        output_tokens = self._coerce_int(usage.get("completion_tokens"))
        total_tokens = self._coerce_int(usage.get("total_tokens"))
        if output_tokens is None and prompt_tokens is not None and total_tokens is not None:
            output_tokens = max(0, total_tokens - prompt_tokens)
        return prompt_tokens, output_tokens

    def _log_unsupported_decoding(self, request: ResponseRequest) -> None:
        ignored: dict[str, object] = {}
        if request.decoding.beam_size is not None:
            ignored["beam_size"] = request.decoding.beam_size
        if request.decoding.top_k is not None:
            ignored["top_k"] = request.decoding.top_k
        if request.decoding.repetition_penalty is not None:
            ignored["repetition_penalty"] = request.decoding.repetition_penalty
        if not ignored:
            return
        LOGGER.info(
            "%s",
            json.dumps(
                {
                    "event": "llm_pool.remote_unsupported_decoding",
                    "ignored": ignored,
                    "model": request.model,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )

    def _resolve_decoding(self, request_decoding: DecodingParams) -> ResolvedDecoding:
        defaults = self.decoding_defaults
        return ResolvedDecoding(
            beam_size=request_decoding.beam_size if request_decoding.beam_size is not None else defaults.beam_size,
            top_k=request_decoding.top_k if request_decoding.top_k is not None else defaults.top_k,
            top_p=request_decoding.top_p if request_decoding.top_p is not None else defaults.top_p,
            temperature=request_decoding.temperature if request_decoding.temperature is not None else defaults.temperature,
            repetition_penalty=request_decoding.repetition_penalty
            if request_decoding.repetition_penalty is not None
            else defaults.repetition_penalty,
            max_tokens=request_decoding.max_tokens if request_decoding.max_tokens is not None else defaults.max_tokens,
            stop=list(request_decoding.stop) if request_decoding.stop else list(defaults.stop),
        )

    @staticmethod
    def _required_remote_field(value: str | None, field_name: str) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required for openai_compatible models")
        parsed = value.strip()
        if parsed == "":
            raise ValueError(f"{field_name} is required for openai_compatible models")
        return parsed

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None
