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
class VllmServeModelRuntime:
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


class VllmServeEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, VllmServeModelRuntime] = {}
        self._load_errors: dict[str, str] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_name, model_settings)
            except Exception as exc:
                self._load_errors[model_name] = _exception_message(exc)
                LOGGER.exception(
                    "Failed to start vllm serve for model '%s' from '%s'; skipping model.",
                    model_name,
                    model_settings.vllm_model or model_settings.model_path,
                )
        if not self._models:
            if self._load_errors:
                details = "; ".join(
                    f"{model_name}: {message}"
                    for model_name, message in sorted(self._load_errors.items())
                )
                raise ValueError(details)
            raise ValueError("no enabled vllm_serve models could be loaded")

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
    ) -> VllmServeModelRuntime:
        model_ref = self._model_ref(settings)
        if settings.vllm_serve_timeout_s <= 0.0:
            raise ValueError("vllm_serve_timeout_s must be greater than 0")
        if settings.vllm_serve_start_timeout_s <= 0.0:
            raise ValueError("vllm_serve_start_timeout_s must be greater than 0")
        if settings.vllm_serve_stop_timeout_s < 0.0:
            raise ValueError("vllm_serve_stop_timeout_s must be greater than or equal to 0")

        host = self._required_field(settings.vllm_serve_host, "vllm_serve_host")
        port = settings.vllm_serve_port or self._pick_free_port(host)
        remote_model = settings.vllm_serve_model_alias or model_name
        base_url = f"http://{host}:{port}/v1"
        runtime = VllmServeModelRuntime(
            config=settings,
            process=self._start_process(
                self._command(
                    settings=settings,
                    model_ref=model_ref,
                    host=host,
                    port=port,
                    remote_model=remote_model,
                ),
                settings=settings,
            ),
            base_url=base_url,
            health_url=f"{base_url}/models",
            remote_model=remote_model,
            timeout_s=settings.vllm_serve_timeout_s,
            api_key=settings.vllm_serve_api_key,
            stop_timeout_s=settings.vllm_serve_stop_timeout_s,
        )
        try:
            self._wait_until_ready(runtime, settings.vllm_serve_start_timeout_s)
        except Exception:
            runtime.close()
            raise
        return runtime

    def _command(
        self,
        *,
        settings: ModelSettings,
        model_ref: str,
        host: str,
        port: int,
        remote_model: str,
    ) -> list[str]:
        command = [
            settings.vllm_serve_binary,
            "serve",
            model_ref,
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            remote_model,
            "--dtype",
            settings.vllm_dtype,
            "--tensor-parallel-size",
            str(settings.vllm_tensor_parallel_size),
        ]
        if settings.vllm_gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization", str(settings.vllm_gpu_memory_utilization)])
        if settings.vllm_kv_cache_memory_bytes is not None:
            command.extend(["--kv-cache-memory-bytes", str(settings.vllm_kv_cache_memory_bytes)])
        if settings.vllm_kv_cache_dtype and settings.vllm_kv_cache_dtype != "auto":
            command.extend(["--kv-cache-dtype", settings.vllm_kv_cache_dtype])
        if settings.vllm_max_model_len is not None:
            command.extend(["--max-model-len", str(settings.vllm_max_model_len)])
        if settings.vllm_trust_remote_code:
            command.append("--trust-remote-code")
        if settings.vllm_enforce_eager:
            command.append("--enforce-eager")
        if settings.vllm_limit_mm_per_prompt:
            command.extend([
                "--limit-mm-per-prompt",
                json.dumps(dict(settings.vllm_limit_mm_per_prompt), ensure_ascii=True),
            ])
        if settings.vllm_mm_processor_kwargs:
            command.extend([
                "--mm-processor-kwargs",
                json.dumps(dict(settings.vllm_mm_processor_kwargs), ensure_ascii=True),
            ])
        if settings.vllm_speculative_method:
            speculative_config: dict[str, object] = {
                "method": settings.vllm_speculative_method,
                "num_speculative_tokens": settings.vllm_num_speculative_tokens,
            }
            if settings.vllm_speculative_model is not None:
                speculative_config["model"] = settings.vllm_speculative_model
            command.extend([
                "--speculative-config",
                json.dumps(speculative_config, ensure_ascii=True),
            ])
        if settings.vllm_serve_api_key is not None:
            command.extend(["--api-key", settings.vllm_serve_api_key])
        command.extend(settings.vllm_serve_extra_args)
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
            raise RuntimeError(f"vllm serve binary not found: {command[0]}") from exc

    @staticmethod
    def _subprocess_env(settings: ModelSettings) -> dict[str, str] | None:
        if not settings.vllm_serve_library_path and not settings.vllm_serve_env:
            return None
        env = os.environ.copy()
        if settings.vllm_serve_library_path:
            existing_library_path = env.get("LD_LIBRARY_PATH", "")
            path_items = [*settings.vllm_serve_library_path]
            if existing_library_path:
                path_items.append(existing_library_path)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(path_items)
        for key, value in settings.vllm_serve_env:
            env[key] = value
        return env

    def _wait_until_ready(
        self,
        runtime: VllmServeModelRuntime,
        start_timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + start_timeout_s
        last_error = "models endpoint did not respond"
        while time.monotonic() < deadline:
            return_code = runtime.process.poll()
            if return_code is not None:
                raise RuntimeError(f"vllm serve exited during startup with code {return_code}")
            try:
                self._get_health(runtime)
                return
            except Exception as exc:
                last_error = _exception_message(exc)
            time.sleep(0.25)
        raise RuntimeError(f"vllm serve did not become ready within {start_timeout_s:g}s: {last_error}")

    def _get_health(self, runtime: VllmServeModelRuntime) -> None:
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
        runtime: VllmServeModelRuntime,
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
        runtime: VllmServeModelRuntime,
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
                code="vllm_serve_timeout",
                status_code=504,
                message="vllm serve chat completion timed out",
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BackendExecutionError(
                    code="vllm_serve_timeout",
                    status_code=504,
                    message="vllm serve chat completion timed out",
                ) from exc
            raise BackendExecutionError(
                code="vllm_serve_connection_error",
                status_code=502,
                message="vllm serve chat completion connection failed",
            ) from exc

        try:
            parsed = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendExecutionError(
                code="vllm_serve_response_parse_failure",
                status_code=502,
                message="vllm serve chat completion returned invalid JSON",
            ) from exc
        if not isinstance(parsed, dict):
            raise BackendExecutionError(
                code="vllm_serve_response_parse_failure",
                status_code=502,
                message="vllm serve chat completion returned a non-object JSON response",
            )
        return parsed

    def _headers(self, runtime: VllmServeModelRuntime) -> dict[str, str]:
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
                code="vllm_serve_authentication_failure",
                status_code=502,
                message=f"vllm serve chat completion authentication failed with HTTP {status}",
            )
        if 400 <= status < 500:
            return BackendExecutionError(
                code="vllm_serve_invalid_request",
                status_code=502,
                message=f"vllm serve chat completion rejected the request with HTTP {status}",
            )
        if status >= 500:
            return BackendExecutionError(
                code="vllm_serve_error",
                status_code=502,
                message=f"vllm serve chat completion failed with HTTP {status}",
            )
        return BackendExecutionError(
            code="vllm_serve_http_error",
            status_code=502,
            message=f"vllm serve chat completion failed with HTTP {status}",
        )

    def _extract_text(self, payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BackendExecutionError(
                code="vllm_serve_response_parse_failure",
                status_code=502,
                message="vllm serve chat completion response did not contain choices",
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise BackendExecutionError(
                code="vllm_serve_response_parse_failure",
                status_code=502,
                message="vllm serve chat completion choice was not an object",
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise BackendExecutionError(
                code="vllm_serve_response_parse_failure",
                status_code=502,
                message="vllm serve chat completion choice did not contain a message",
            )
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        raise BackendExecutionError(
            code="vllm_serve_response_parse_failure",
            status_code=502,
            message="vllm serve chat completion message content was not text",
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
                    "event": "llm_pool.vllm_serve_unsupported_decoding",
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
    def _model_ref(settings: ModelSettings) -> str:
        model_ref = (settings.vllm_model or settings.model_path or "").strip()
        if model_ref == "":
            raise ValueError("vllm_serve backend requires model_path or vllm_model to be set")
        return model_ref

    @staticmethod
    def _required_field(value: str | None, field_name: str) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required for vllm_serve models")
        parsed = value.strip()
        if parsed == "":
            raise ValueError(f"{field_name} is required for vllm_serve models")
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
