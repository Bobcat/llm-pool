from __future__ import annotations

from dataclasses import dataclass
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from typing import BinaryIO
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import AudioContent
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
from .common import _resolve_request_enable_thinking


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Return only the response."


@dataclass
class TrtllmServeModelRuntime:
    config: ModelSettings
    process: subprocess.Popen
    base_url: str
    health_url: str
    remote_model: str
    timeout_s: float
    stop_timeout_s: float
    output_log: BinaryIO

    def close(self) -> None:
        if self.output_log.closed:
            return
        try:
            leader_running = self.process.poll() is None
            try:
                # start_new_session=True makes the child the process-group leader.
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if not leader_running:
                deadline = time.monotonic() + self.stop_timeout_s
                while time.monotonic() < deadline:
                    try:
                        os.killpg(self.process.pid, 0)
                    except ProcessLookupError:
                        return
                    time.sleep(0.1)
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
            try:
                self.process.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    LOGGER.warning(
                        "TensorRT-LLM serve process group %s did not exit after SIGKILL",
                        self.process.pid,
                    )
        finally:
            self.output_log.close()

    def output_tail(self, max_bytes: int = 16 * 1024) -> str:
        try:
            file_descriptor = self.output_log.fileno()
            file_size = os.fstat(file_descriptor).st_size
            raw_output = os.pread(
                file_descriptor,
                max_bytes,
                max(0, file_size - max_bytes),
            )
        except (OSError, ValueError):
            return ""
        return raw_output.decode("utf-8", errors="replace").strip()


class TrtllmServeEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, TrtllmServeModelRuntime] = {}
        self._load_errors: dict[str, str] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_name, model_settings)
            except Exception as exc:
                self._load_errors[model_name] = _exception_message(exc)
                LOGGER.exception(
                    "Failed to start TensorRT-LLM serve for model '%s' from '%s'; skipping model.",
                    model_name,
                    model_settings.trtllm_model or model_settings.model_path,
                )
        if not self._models:
            if self._load_errors:
                details = "; ".join(
                    f"{model_name}: {message}"
                    for model_name, message in sorted(self._load_errors.items())
                )
                raise ValueError(details)
            raise ValueError("no enabled trtllm_serve models could be loaded")

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
    ) -> TrtllmServeModelRuntime:
        model_ref = self._model_ref(settings)
        if settings.trtllm_serve_timeout_s <= 0.0:
            raise ValueError("trtllm_serve_timeout_s must be greater than 0")
        if settings.trtllm_serve_start_timeout_s <= 0.0:
            raise ValueError("trtllm_serve_start_timeout_s must be greater than 0")
        if settings.trtllm_serve_stop_timeout_s < 0.0:
            raise ValueError("trtllm_serve_stop_timeout_s must be greater than or equal to 0")

        host = self._required_field(settings.trtllm_serve_host, "trtllm_serve_host")
        port = settings.trtllm_serve_port or self._pick_free_port(host)
        remote_model = settings.trtllm_serve_model_alias or model_name
        base_url = f"http://{host}:{port}/v1"
        process, output_log = self._start_process(
            self._command(
                settings=settings,
                model_ref=model_ref,
                host=host,
                port=port,
                remote_model=remote_model,
            ),
            settings=settings,
        )
        runtime = TrtllmServeModelRuntime(
            config=settings,
            process=process,
            base_url=base_url,
            health_url=f"http://{host}:{port}/health",
            remote_model=remote_model,
            timeout_s=settings.trtllm_serve_timeout_s,
            stop_timeout_s=settings.trtllm_serve_stop_timeout_s,
            output_log=output_log,
        )
        try:
            self._wait_until_ready(runtime, settings.trtllm_serve_start_timeout_s)
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
            settings.trtllm_serve_binary,
            "serve",
            model_ref,
            "--host",
            host,
            "--port",
            str(port),
            "--served_model_name",
            remote_model,
            "--no-telemetry",
        ]
        if settings.trtllm_serve_config_path is not None:
            command.extend(["--config", settings.trtllm_serve_config_path])
        if settings.trtllm_serve_reasoning_parser is not None:
            command.extend([
                "--reasoning_parser",
                settings.trtllm_serve_reasoning_parser,
            ])
        if settings.trtllm_serve_tool_parser is not None:
            command.extend(["--tool_parser", settings.trtllm_serve_tool_parser])
        if settings.trtllm_trust_remote_code:
            command.append("--trust_remote_code")
        command.extend(settings.trtllm_serve_extra_args)
        return command

    def _start_process(
        self,
        command: list[str],
        *,
        settings: ModelSettings,
    ) -> tuple[subprocess.Popen, BinaryIO]:
        output_log = tempfile.TemporaryFile(
            prefix="llm-pool-trtllm-serve-",
            suffix=".log",
        )
        try:
            process = subprocess.Popen(
                command,
                env=self._subprocess_env(settings),
                stdout=output_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            output_log.close()
            raise RuntimeError(f"TensorRT-LLM serve binary not found: {command[0]}") from exc
        except Exception:
            output_log.close()
            raise
        return process, output_log

    @staticmethod
    def _subprocess_env(settings: ModelSettings) -> dict[str, str]:
        binary_dir = os.path.dirname(settings.trtllm_serve_binary)
        env = os.environ.copy()
        for key, value in settings.trtllm_serve_env:
            env[key] = value
        env["PYTHONUNBUFFERED"] = "1"
        path_items = [binary_dir]
        cuda_home = env.get("CUDA_HOME")
        if cuda_home:
            path_items.append(os.path.join(cuda_home, "bin"))
        existing_path = env.get("PATH", "")
        if existing_path:
            path_items.append(existing_path)
        if any(path_items):
            env["PATH"] = os.pathsep.join(item for item in path_items if item)
        if settings.trtllm_serve_library_path:
            existing_library_path = env.get("LD_LIBRARY_PATH", "")
            path_items = [*settings.trtllm_serve_library_path]
            if existing_library_path:
                path_items.append(existing_library_path)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(path_items)
        return env

    def _wait_until_ready(
        self,
        runtime: TrtllmServeModelRuntime,
        start_timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + start_timeout_s
        last_error = "health endpoint did not respond"
        while time.monotonic() < deadline:
            return_code = runtime.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    self._startup_failure_message(
                        runtime,
                        f"TensorRT-LLM serve exited during startup with code {return_code}",
                    )
                )
            try:
                self._get_health(runtime)
                return
            except Exception as exc:
                last_error = _exception_message(exc)
            time.sleep(0.25)
        raise RuntimeError(
            self._startup_failure_message(
                runtime,
                "TensorRT-LLM serve did not become ready within "
                f"{start_timeout_s:g}s: {last_error}",
            )
        )

    @staticmethod
    def _startup_failure_message(
        runtime: TrtllmServeModelRuntime,
        message: str,
    ) -> str:
        output_tail = runtime.output_tail()
        if output_tail == "":
            return message
        return f"{message}\nTensorRT-LLM output tail:\n{output_tail}"

    @staticmethod
    def _get_health(runtime: TrtllmServeModelRuntime) -> None:
        request = Request(runtime.health_url, method="GET")
        with urlopen(request, timeout=1.0) as response:
            response.read()

    def _chat_completions_payload(
        self,
        *,
        runtime: TrtllmServeModelRuntime,
        request: ResponseRequest,
        decoding: ResolvedDecoding,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": runtime.remote_model,
            "messages": self._chat_messages(request),
            "temperature": decoding.temperature,
            "top_k": decoding.top_k,
            "top_p": decoding.top_p,
            "repetition_penalty": decoding.repetition_penalty,
            "max_tokens": decoding.max_tokens,
        }
        if decoding.stop:
            payload["stop"] = decoding.stop
        enable_thinking = _resolve_request_enable_thinking(
            request,
            runtime.config.enable_thinking,
        )
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking,
            }
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
        content: str | list[TextContent | ImageContent | AudioContent] | None,
    ) -> str | list[dict[str, object]]:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return [self._content_item_payload(item) for item in content]

    @staticmethod
    def _content_item_payload(
        item: TextContent | ImageContent | AudioContent,
    ) -> dict[str, object]:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="python")
        return item.dict()

    def _post_json(
        self,
        runtime: TrtllmServeModelRuntime,
        payload: dict[str, object],
    ) -> dict[str, object]:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = Request(
            f"{runtime.base_url}/chat/completions",
            data=data,
            headers={
                "Accept": "application/json",
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
                code="trtllm_serve_timeout",
                status_code=504,
                message="TensorRT-LLM serve chat completion timed out",
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BackendExecutionError(
                    code="trtllm_serve_timeout",
                    status_code=504,
                    message="TensorRT-LLM serve chat completion timed out",
                ) from exc
            raise BackendExecutionError(
                code="trtllm_serve_connection_error",
                status_code=502,
                message="TensorRT-LLM serve chat completion connection failed",
            ) from exc

        try:
            parsed = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendExecutionError(
                code="trtllm_serve_response_parse_failure",
                status_code=502,
                message="TensorRT-LLM serve chat completion returned invalid JSON",
            ) from exc
        if not isinstance(parsed, dict):
            raise BackendExecutionError(
                code="trtllm_serve_response_parse_failure",
                status_code=502,
                message=(
                    "TensorRT-LLM serve chat completion returned a non-object JSON response"
                ),
            )
        return parsed

    @staticmethod
    def _map_http_error(exc: HTTPError) -> BackendExecutionError:
        status = int(exc.code)
        if 400 <= status < 500:
            return BackendExecutionError(
                code="trtllm_serve_invalid_request",
                status_code=502,
                message=(
                    "TensorRT-LLM serve chat completion rejected the request "
                    f"with HTTP {status}"
                ),
            )
        if status >= 500:
            return BackendExecutionError(
                code="trtllm_serve_error",
                status_code=502,
                message=(
                    "TensorRT-LLM serve chat completion failed "
                    f"with HTTP {status}"
                ),
            )
        return BackendExecutionError(
            code="trtllm_serve_http_error",
            status_code=502,
            message=f"TensorRT-LLM serve chat completion failed with HTTP {status}",
        )

    @staticmethod
    def _extract_text(payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BackendExecutionError(
                code="trtllm_serve_response_parse_failure",
                status_code=502,
                message="TensorRT-LLM serve response did not contain choices",
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise BackendExecutionError(
                code="trtllm_serve_response_parse_failure",
                status_code=502,
                message="TensorRT-LLM serve response choice was not an object",
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise BackendExecutionError(
                code="trtllm_serve_response_parse_failure",
                status_code=502,
                message="TensorRT-LLM serve response choice did not contain a message",
            )
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
            reasoning_content = message.get("reasoning_content")
            if (
                text == ""
                and isinstance(reasoning_content, str)
                and reasoning_content.strip() != ""
            ):
                raise BackendExecutionError(
                    code="trtllm_serve_incomplete_response",
                    status_code=502,
                    message=(
                        "TensorRT-LLM serve response ended before producing final content"
                    ),
                )
            return text
        raise BackendExecutionError(
            code="trtllm_serve_response_parse_failure",
            status_code=502,
            message="TensorRT-LLM serve response message content was not text",
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

    @staticmethod
    def _log_unsupported_decoding(request: ResponseRequest) -> None:
        if request.decoding.beam_size is None:
            return
        LOGGER.info(
            "%s",
            json.dumps(
                {
                    "event": "llm_pool.trtllm_serve_unsupported_decoding",
                    "ignored": {"beam_size": request.decoding.beam_size},
                    "model": request.model,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )

    def _resolve_decoding(self, request_decoding: DecodingParams) -> ResolvedDecoding:
        defaults = self.decoding_defaults
        return ResolvedDecoding(
            beam_size=(
                request_decoding.beam_size
                if request_decoding.beam_size is not None
                else defaults.beam_size
            ),
            top_k=(
                request_decoding.top_k
                if request_decoding.top_k is not None
                else defaults.top_k
            ),
            top_p=(
                request_decoding.top_p
                if request_decoding.top_p is not None
                else defaults.top_p
            ),
            temperature=(
                request_decoding.temperature
                if request_decoding.temperature is not None
                else defaults.temperature
            ),
            repetition_penalty=(
                request_decoding.repetition_penalty
                if request_decoding.repetition_penalty is not None
                else defaults.repetition_penalty
            ),
            max_tokens=(
                request_decoding.max_tokens
                if request_decoding.max_tokens is not None
                else defaults.max_tokens
            ),
            stop=(
                list(request_decoding.stop)
                if request_decoding.stop
                else list(defaults.stop)
            ),
        )

    @staticmethod
    def _model_ref(settings: ModelSettings) -> str:
        model_ref = (settings.trtllm_model or settings.model_path or "").strip()
        if model_ref == "":
            raise ValueError(
                "trtllm_serve backend requires model_path or trtllm_model to be set"
            )
        return model_ref

    @staticmethod
    def _required_field(value: str | None, field_name: str) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required for trtllm_serve models")
        parsed = value.strip()
        if parsed == "":
            raise ValueError(f"{field_name} is required for trtllm_serve models")
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
