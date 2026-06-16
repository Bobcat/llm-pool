from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import DecodingParams
from app.schemas import EngineResult
from app.schemas import ImageContent
from app.schemas import Message
from app.schemas import ResponseMetrics
from app.schemas import ResponseRequest
from app.schemas import TextContent

from .common import BackendExecutionError
from .common import LOGGER
from .common import ResolvedDecoding
from .common import _exception_message


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Return only the response."


@dataclass
class LlamaServerModelRuntime:
    config: ModelSettings
    process: subprocess.Popen
    base_url: str
    health_url: str
    remote_model: str
    timeout_s: float
    api_key: str | None
    stop_timeout_s: float

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=self.stop_timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)


class LlamaServerEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, LlamaServerModelRuntime] = {}
        self._load_errors: dict[str, str] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_name, model_settings)
            except Exception as exc:
                self._load_errors[model_name] = _exception_message(exc)
                LOGGER.exception(
                    "Failed to start llama-server for model '%s' from '%s'; skipping model.",
                    model_name,
                    model_settings.model_path,
                )
        if not self._models:
            if self._load_errors:
                details = "; ".join(
                    f"{model_name}: {message}"
                    for model_name, message in sorted(self._load_errors.items())
                )
                raise ValueError(details)
            raise ValueError("no enabled llama-server models could be loaded")

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
                backend_inference_wall_ms=wall_s * 1000.0,
                engine_prompt_tokens=prompt_tokens,
                engine_output_tokens=output_tokens,
                engine_tokens_per_second=tokens_per_second,
            ),
        )

    def _build_runtime(
        self,
        model_name: str,
        settings: ModelSettings,
    ) -> LlamaServerModelRuntime:
        model_path = self._required_field(settings.model_path, "model_path")
        if settings.llama_server_timeout_s <= 0.0:
            raise ValueError("llama_server_timeout_s must be greater than 0")
        if settings.llama_server_start_timeout_s <= 0.0:
            raise ValueError("llama_server_start_timeout_s must be greater than 0")
        if settings.llama_server_stop_timeout_s < 0.0:
            raise ValueError("llama_server_stop_timeout_s must be greater than or equal to 0")

        host = self._required_field(settings.llama_server_host, "llama_server_host")
        port = settings.llama_server_port or self._pick_free_port(host)
        remote_model = settings.llama_server_model_alias or model_name
        base_url = f"http://{host}:{port}/v1"
        runtime = LlamaServerModelRuntime(
            config=settings,
            process=self._start_process(
                self._command(
                    settings=settings,
                    model_path=model_path,
                    host=host,
                    port=port,
                    remote_model=remote_model,
                ),
                settings=settings,
            ),
            base_url=base_url,
            health_url=f"http://{host}:{port}/health",
            remote_model=remote_model,
            timeout_s=settings.llama_server_timeout_s,
            api_key=settings.llama_server_api_key,
            stop_timeout_s=settings.llama_server_stop_timeout_s,
        )
        try:
            self._wait_until_ready(runtime, settings.llama_server_start_timeout_s)
        except Exception:
            runtime.close()
            raise
        return runtime

    def _command(
        self,
        *,
        settings: ModelSettings,
        model_path: str,
        host: str,
        port: int,
        remote_model: str,
    ) -> list[str]:
        command = [
            settings.llama_server_binary,
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--alias",
            remote_model,
            "--no-ui",
            "-fa",
            settings.llama_server_flash_attn,
        ]
        if settings.llama_server_n_ctx is not None:
            command.extend(["-c", str(settings.llama_server_n_ctx)])
        if settings.llama_server_n_gpu_layers is not None:
            command.extend(["-ngl", settings.llama_server_n_gpu_layers])
        if settings.llama_server_mmproj_path is not None:
            command.extend(["--mmproj", settings.llama_server_mmproj_path])
        if settings.llama_server_image_max_tokens is not None:
            command.extend(["--image-max-tokens", str(settings.llama_server_image_max_tokens)])
        if settings.llama_server_draft_model_path is not None:
            command.extend(["--model-draft", settings.llama_server_draft_model_path])
        if settings.llama_server_spec_type is not None:
            command.extend(["--spec-type", settings.llama_server_spec_type])
        if settings.llama_server_spec_draft_n_max is not None:
            command.extend(["--spec-draft-n-max", str(settings.llama_server_spec_draft_n_max)])
        if settings.llama_server_spec_draft_p_min is not None:
            command.extend(["--spec-draft-p-min", str(settings.llama_server_spec_draft_p_min)])
        if settings.llama_server_spec_draft_ngl is not None:
            command.extend(["--spec-draft-ngl", settings.llama_server_spec_draft_ngl])
        if settings.llama_server_reasoning is not None:
            command.extend(["--reasoning", settings.llama_server_reasoning])
        if settings.llama_server_api_key is not None:
            command.extend(["--api-key", settings.llama_server_api_key])
        command.extend(settings.llama_server_extra_args)
        return command

    def _start_process(self, command: list[str], *, settings: ModelSettings) -> subprocess.Popen:
        try:
            return subprocess.Popen(
                command,
                env=self._subprocess_env(settings),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"llama-server binary not found: {command[0]}") from exc

    @staticmethod
    def _subprocess_env(settings: ModelSettings) -> dict[str, str] | None:
        if not settings.llama_server_library_path:
            return None
        env = os.environ.copy()
        existing_library_path = env.get("LD_LIBRARY_PATH", "")
        path_items = [*settings.llama_server_library_path]
        if existing_library_path:
            path_items.append(existing_library_path)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(path_items)
        return env

    def _wait_until_ready(
        self,
        runtime: LlamaServerModelRuntime,
        start_timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + start_timeout_s
        last_error = "health endpoint did not respond"
        while time.monotonic() < deadline:
            return_code = runtime.process.poll()
            if return_code is not None:
                raise RuntimeError(f"llama-server exited during startup with code {return_code}")
            try:
                self._get_health(runtime)
                return
            except Exception as exc:
                last_error = _exception_message(exc)
            time.sleep(0.25)
        raise RuntimeError(f"llama-server did not become ready within {start_timeout_s:g}s: {last_error}")

    def _get_health(self, runtime: LlamaServerModelRuntime) -> None:
        request = Request(
            runtime.health_url,
            headers=self._headers(runtime),
            method="GET",
        )
        with urlopen(request, timeout=1.0) as response:
            response.read()

    def _chat_completions_payload(
        self,
        *,
        runtime: LlamaServerModelRuntime,
        request: ResponseRequest,
        decoding: ResolvedDecoding,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": runtime.remote_model,
            "messages": self._chat_messages(request),
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "max_tokens": decoding.max_tokens,
        }
        if decoding.stop:
            payload["stop"] = decoding.stop
        return payload

    def _chat_messages(self, request: ResponseRequest) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": request.instructions or _DEFAULT_SYSTEM_PROMPT}
        ]
        if request.messages is not None:
            for message in request.messages:
                messages.append(self._message_payload(message))
            return messages
        messages.append(
            {
                "role": "user",
                "content": self._content_payload(request.input),
            }
        )
        return messages

    def _message_payload(self, message: Message) -> dict[str, object]:
        return {
            "role": message.role,
            "content": self._content_payload(message.content),
        }

    def _content_payload(
        self,
        content: str | list[TextContent | ImageContent] | None,
    ) -> str | list[dict[str, object]]:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return [self._content_item_payload(item) for item in content]

    @staticmethod
    def _content_item_payload(item: TextContent | ImageContent) -> dict[str, object]:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="python")
        return item.dict()

    def _post_json(
        self,
        runtime: LlamaServerModelRuntime,
        payload: dict[str, object],
    ) -> dict[str, object]:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = Request(
            f"{runtime.base_url}/chat/completions",
            data=data,
            headers=self._headers(runtime),
            method="POST",
        )
        try:
            with urlopen(request, timeout=runtime.timeout_s) as response:
                raw_payload = response.read()
        except HTTPError as exc:
            raise self._map_http_error(exc) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise BackendExecutionError(
                code="llama_server_timeout",
                status_code=504,
                message="llama-server chat completion timed out",
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BackendExecutionError(
                    code="llama_server_timeout",
                    status_code=504,
                    message="llama-server chat completion timed out",
                ) from exc
            raise BackendExecutionError(
                code="llama_server_connection_error",
                status_code=502,
                message="llama-server chat completion connection failed",
            ) from exc

        try:
            parsed = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendExecutionError(
                code="llama_server_response_parse_failure",
                status_code=502,
                message="llama-server chat completion returned invalid JSON",
            ) from exc
        if not isinstance(parsed, dict):
            raise BackendExecutionError(
                code="llama_server_response_parse_failure",
                status_code=502,
                message="llama-server chat completion returned a non-object JSON response",
            )
        return parsed

    def _headers(self, runtime: LlamaServerModelRuntime) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if runtime.api_key is not None:
            headers["Authorization"] = f"Bearer {runtime.api_key}"
        return headers

    def _map_http_error(self, exc: HTTPError) -> BackendExecutionError:
        status = int(exc.code)
        if status in {401, 403}:
            return BackendExecutionError(
                code="llama_server_authentication_failure",
                status_code=502,
                message=f"llama-server chat completion authentication failed with HTTP {status}",
            )
        if 400 <= status < 500:
            return BackendExecutionError(
                code="llama_server_invalid_request",
                status_code=502,
                message=f"llama-server chat completion rejected the request with HTTP {status}",
            )
        if status >= 500:
            return BackendExecutionError(
                code="llama_server_error",
                status_code=502,
                message=f"llama-server chat completion failed with HTTP {status}",
            )
        return BackendExecutionError(
            code="llama_server_http_error",
            status_code=502,
            message=f"llama-server chat completion failed with HTTP {status}",
        )

    def _extract_text(self, payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BackendExecutionError(
                code="llama_server_response_parse_failure",
                status_code=502,
                message="llama-server chat completion response did not contain choices",
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise BackendExecutionError(
                code="llama_server_response_parse_failure",
                status_code=502,
                message="llama-server chat completion choice was not an object",
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise BackendExecutionError(
                code="llama_server_response_parse_failure",
                status_code=502,
                message="llama-server chat completion choice did not contain a message",
            )
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        raise BackendExecutionError(
            code="llama_server_response_parse_failure",
            status_code=502,
            message="llama-server chat completion message content was not text",
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
                    "event": "llm_pool.llama_server_unsupported_decoding",
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
    def _required_field(value: str | None, field_name: str) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required for llama_server models")
        parsed = value.strip()
        if parsed == "":
            raise ValueError(f"{field_name} is required for llama_server models")
        return parsed

    @staticmethod
    def _pick_free_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None
