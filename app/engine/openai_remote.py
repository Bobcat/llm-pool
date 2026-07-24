from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import os
import socket
import time
import uuid
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request
from urllib.request import urlopen

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import DecodingParams
from app.schemas import EngineResult
from app.schemas import FileContent
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
_REMOTE_FILE_MODES = {"chat_completions_inline", "files_extract"}


@dataclass(frozen=True)
class OpenAIRemoteModelRuntime:
    config: ModelSettings
    base_url: str
    api_key_env: str
    remote_model: str
    timeout_s: float


class OpenAIRemoteEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, OpenAIRemoteModelRuntime] = {}
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
        extracted_file_contents = self._extract_file_contents(runtime, request)
        payload = self._chat_completions_payload(
            runtime=runtime,
            request=request,
            decoding=decoding,
            extracted_file_contents=extracted_file_contents,
        )

        started = time.perf_counter()
        response_payload = self._post_json(runtime, payload)
        wall_s = max(0.0, time.perf_counter() - started)

        text = self._extract_text(response_payload)
        prompt_tokens, output_tokens, cached_prompt_tokens = self._extract_usage(
            response_payload
        )
        tokens_per_second = None
        if output_tokens is not None and wall_s > 0.0:
            tokens_per_second = output_tokens / wall_s
        return EngineResult(
            text=text,
            metrics=ResponseMetrics(
                engine_prompt_tokens=prompt_tokens,
                engine_cached_prompt_tokens=cached_prompt_tokens,
                engine_output_tokens=output_tokens,
                engine_tokens_per_second=tokens_per_second,
            ),
        )

    def _build_runtime(self, settings: ModelSettings) -> OpenAIRemoteModelRuntime:
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
        if (
            settings.remote_file_mode is not None
            and settings.remote_file_mode not in _REMOTE_FILE_MODES
        ):
            raise ValueError(
                "remote_file_mode must be 'chat_completions_inline', "
                "'files_extract', or omitted"
            )
        if (
            settings.remote_file_mode == "files_extract"
            and settings.remote_file_purpose is None
        ):
            raise ValueError(
                "remote_file_purpose is required when remote_file_mode is 'files_extract'"
            )

        base_url = self._required_remote_field(settings.remote_base_url, "remote_base_url").rstrip("/")
        api_key_env = self._required_remote_field(settings.remote_api_key_env, "remote_api_key_env")
        remote_model = self._required_remote_field(settings.remote_model, "remote_model")
        if os.environ.get(api_key_env, "").strip() == "":
            raise RuntimeError(f"missing API key environment variable: {api_key_env}")

        return OpenAIRemoteModelRuntime(
            config=settings,
            base_url=base_url,
            api_key_env=api_key_env,
            remote_model=remote_model,
            timeout_s=settings.remote_timeout_s,
        )

    def _chat_completions_payload(
        self,
        *,
        runtime: OpenAIRemoteModelRuntime,
        request: ResponseRequest,
        decoding: ResolvedDecoding,
        extracted_file_contents: list[str] | None = None,
    ) -> dict[str, object]:
        system_prompt = request.instructions or _DEFAULT_SYSTEM_PROMPT
        include_files = runtime.config.remote_file_mode != "files_extract"
        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
        ]
        for file_content in extracted_file_contents or []:
            messages.append({"role": "system", "content": file_content})
        if request.messages is not None:
            messages.extend(
                {
                    "role": message.role,
                    "content": self._message_content(
                        message.content,
                        include_files=include_files,
                    ),
                }
                for message in request.messages
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": self._message_content(
                        request.input,
                        include_files=include_files,
                    ),
                }
            )
        payload: dict[str, object] = {
            "model": runtime.remote_model,
            "messages": messages,
            "max_tokens": decoding.max_tokens,
        }
        remote_thinking = _resolve_request_remote_thinking(
            request,
            runtime.config.remote_thinking,
        )
        if self._include_temperature(
            remote_model=runtime.remote_model,
            remote_thinking=remote_thinking,
            temperature=decoding.temperature,
        ):
            payload["temperature"] = decoding.temperature
        if self._include_top_p(remote_model=runtime.remote_model, top_p=decoding.top_p):
            payload["top_p"] = decoding.top_p
        if decoding.stop:
            payload["stop"] = decoding.stop
        if remote_thinking is not None:
            payload["thinking"] = {"type": remote_thinking}
        if (
            runtime.config.remote_prompt_cache_key_enabled
            and request.prompt_cache_key is not None
        ):
            payload["prompt_cache_key"] = request.prompt_cache_key
        return payload

    @staticmethod
    def _include_temperature(
        *,
        remote_model: str,
        remote_thinking: str | None,
        temperature: float,
    ) -> bool:
        if not remote_model.strip().lower().startswith("kimi-"):
            return True
        if remote_thinking == "disabled":
            return temperature == 0.6
        return temperature == 1.0

    @staticmethod
    def _include_top_p(*, remote_model: str, top_p: float) -> bool:
        if not remote_model.strip().lower().startswith("kimi-"):
            return True
        return top_p == 0.95

    @staticmethod
    def _message_content(
        content,
        *,
        include_files: bool,
    ) -> str | list[dict[str, object]]:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return [
            item.model_dump(mode="python")
            for item in content
            if include_files or not isinstance(item, FileContent)
        ]

    def _extract_file_contents(
        self,
        runtime: OpenAIRemoteModelRuntime,
        request: ResponseRequest,
    ) -> list[str]:
        file_items = self._request_file_items(request)
        if not file_items:
            return []
        if runtime.config.remote_file_mode != "files_extract":
            return []

        extracted: list[str] = []
        for item in file_items:
            filename = item.file.filename
            media_type, file_bytes = self._decode_file_data(
                filename=filename,
                file_data=item.file.file_data,
            )
            file_id = self._upload_file(
                runtime,
                filename=filename,
                media_type=media_type,
                file_bytes=file_bytes,
            )
            try:
                extracted.append(self._get_file_content(runtime, file_id))
            finally:
                self._delete_file_best_effort(runtime, file_id)
        return extracted

    @staticmethod
    def _request_file_items(request: ResponseRequest) -> list[FileContent]:
        file_items: list[FileContent] = []
        if isinstance(request.input, list):
            file_items.extend(
                item for item in request.input if isinstance(item, FileContent)
            )
        for message in request.messages or []:
            if isinstance(message.content, list):
                file_items.extend(
                    item
                    for item in message.content
                    if isinstance(item, FileContent)
                )
        return file_items

    @staticmethod
    def _decode_file_data(*, filename: str, file_data: str) -> tuple[str, bytes]:
        header, separator, encoded = file_data.partition(",")
        if (
            separator == ""
            or not header.startswith("data:")
            or ";base64" not in header.lower()
        ):
            raise BackendExecutionError(
                code="invalid_file_input",
                status_code=400,
                message=f"file {filename!r} must use a base64 data URL",
            )
        media_type = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BackendExecutionError(
                code="invalid_file_input",
                status_code=400,
                message=f"file {filename!r} contains invalid base64 data",
            ) from exc
        return media_type, decoded

    def _upload_file(
        self,
        runtime: OpenAIRemoteModelRuntime,
        *,
        filename: str,
        media_type: str,
        file_bytes: bytes,
    ) -> str:
        purpose = runtime.config.remote_file_purpose
        if purpose is None:
            raise BackendExecutionError(
                code="remote_file_configuration_error",
                status_code=500,
                message="remote file extraction purpose is not configured",
            )
        boundary = f"----llm-pool-{uuid.uuid4().hex}"
        safe_filename = filename.replace("\r", "").replace("\n", "").replace('"', "_")
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            f"{purpose}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode("utf-8")
        body += file_bytes
        body += f"\r\n--{boundary}--\r\n".encode("ascii")
        request = Request(
            f"{runtime.base_url}/files",
            data=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key(runtime)}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        raw_payload = self._request_bytes(
            runtime,
            request,
            operation="file upload",
        )
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream file upload returned invalid JSON",
            ) from exc
        file_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(file_id, str) or file_id.strip() == "":
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream file upload response did not contain a file id",
            )
        return file_id.strip()

    def _get_file_content(
        self,
        runtime: OpenAIRemoteModelRuntime,
        file_id: str,
    ) -> str:
        request = Request(
            f"{runtime.base_url}/files/{quote(file_id, safe='')}/content",
            headers={
                "Accept": "text/plain",
                "Authorization": f"Bearer {self._api_key(runtime)}",
            },
            method="GET",
        )
        raw_payload = self._request_bytes(
            runtime,
            request,
            operation="file content extraction",
        )
        try:
            return raw_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BackendExecutionError(
                code="upstream_response_parse_failure",
                status_code=502,
                message="upstream file content was not valid UTF-8 text",
            ) from exc

    def _delete_file_best_effort(
        self,
        runtime: OpenAIRemoteModelRuntime,
        file_id: str,
    ) -> None:
        request = Request(
            f"{runtime.base_url}/files/{quote(file_id, safe='')}",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key(runtime)}",
            },
            method="DELETE",
        )
        try:
            self._request_bytes(
                runtime,
                request,
                operation="file cleanup",
            )
        except BackendExecutionError as exc:
            LOGGER.warning(
                "Failed to delete upstream file %s after extraction: %s",
                file_id,
                exc.message,
            )

    def _request_bytes(
        self,
        runtime: OpenAIRemoteModelRuntime,
        request: Request,
        *,
        operation: str,
    ) -> bytes:
        try:
            with urlopen(request, timeout=runtime.timeout_s) as response:
                return response.read()
        except HTTPError as exc:
            raise self._map_operation_http_error(exc, operation=operation) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise BackendExecutionError(
                code="upstream_timeout",
                status_code=504,
                message=f"upstream {operation} timed out",
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BackendExecutionError(
                    code="upstream_timeout",
                    status_code=504,
                    message=f"upstream {operation} timed out",
                ) from exc
            raise BackendExecutionError(
                code="upstream_connection_error",
                status_code=502,
                message=f"upstream {operation} connection failed",
            ) from exc

    def _map_operation_http_error(
        self,
        exc: HTTPError,
        *,
        operation: str,
    ) -> BackendExecutionError:
        status = int(exc.code)
        upstream_detail = self._http_error_detail(exc)
        detail_suffix = f": {upstream_detail}" if upstream_detail else ""
        if status in {401, 403}:
            return BackendExecutionError(
                code="upstream_authentication_failure",
                status_code=502,
                message=f"upstream {operation} authentication failed with HTTP {status}{detail_suffix}",
            )
        if status == 429:
            return BackendExecutionError(
                code="upstream_rate_limit",
                status_code=429,
                message=f"upstream {operation} was rate limited{detail_suffix}",
            )
        if 400 <= status < 500:
            return BackendExecutionError(
                code="upstream_invalid_request",
                status_code=502,
                message=f"upstream {operation} rejected the request with HTTP {status}{detail_suffix}",
            )
        return BackendExecutionError(
            code="upstream_server_error" if status >= 500 else "upstream_http_error",
            status_code=502,
            message=f"upstream {operation} failed with HTTP {status}{detail_suffix}",
        )

    @staticmethod
    def _api_key(runtime: OpenAIRemoteModelRuntime) -> str:
        api_key = os.environ.get(runtime.api_key_env, "").strip()
        if api_key == "":
            raise BackendExecutionError(
                code="missing_api_key_env",
                status_code=500,
                message=f"missing API key environment variable: {runtime.api_key_env}",
            )
        return api_key

    def _post_json(
        self,
        runtime: OpenAIRemoteModelRuntime,
        payload: dict[str, object],
    ) -> dict[str, object]:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = Request(
            f"{runtime.base_url}/chat/completions",
            data=data,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key(runtime)}",
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
        upstream_detail = self._http_error_detail(exc)
        detail_suffix = f": {upstream_detail}" if upstream_detail else ""
        if status in {401, 403}:
            return BackendExecutionError(
                code="upstream_authentication_failure",
                status_code=502,
                message=f"upstream chat completion authentication failed with HTTP {status}{detail_suffix}",
            )
        if status == 429:
            return BackendExecutionError(
                code="upstream_rate_limit",
                status_code=429,
                message=f"upstream chat completion was rate limited{detail_suffix}",
            )
        if 400 <= status < 500:
            return BackendExecutionError(
                code="upstream_invalid_request",
                status_code=502,
                message=f"upstream chat completion rejected the request with HTTP {status}{detail_suffix}",
            )
        if status >= 500:
            return BackendExecutionError(
                code="upstream_server_error",
                status_code=502,
                message=f"upstream chat completion failed with HTTP {status}{detail_suffix}",
            )
        return BackendExecutionError(
            code="upstream_http_error",
            status_code=502,
            message=f"upstream chat completion failed with HTTP {status}{detail_suffix}",
        )

    @staticmethod
    def _http_error_detail(exc: HTTPError) -> str | None:
        try:
            raw_body = exc.read()
        except Exception:
            return None
        try:
            body = raw_body.decode("utf-8")
        except UnicodeDecodeError:
            return None
        body = body.strip()
        if not body:
            return None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body[:500]
        detail = OpenAIRemoteEngine._http_error_json_detail(parsed)
        if detail is None:
            detail = json.dumps(parsed, ensure_ascii=True)
        detail = detail.strip()
        return detail[:500] if detail else None

    @staticmethod
    def _http_error_json_detail(value: object) -> str | None:
        if isinstance(value, str):
            return value
        if not isinstance(value, dict):
            return None
        error = value.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        for key in ("message", "detail", "error"):
            message = value.get(key)
            if isinstance(message, str):
                return message
        return None

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

    def _extract_usage(
        self,
        payload: dict[str, object],
    ) -> tuple[int | None, int | None, int | None]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None, None, None
        prompt_tokens = self._coerce_int(usage.get("prompt_tokens"))
        output_tokens = self._coerce_int(usage.get("completion_tokens"))
        cached_prompt_tokens = self._coerce_int(usage.get("cached_tokens"))
        total_tokens = self._coerce_int(usage.get("total_tokens"))
        if output_tokens is None and prompt_tokens is not None and total_tokens is not None:
            output_tokens = max(0, total_tokens - prompt_tokens)
        return prompt_tokens, output_tokens, cached_prompt_tokens

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
            raise ValueError(f"{field_name} is required for openai_remote models")
        parsed = value.strip()
        if parsed == "":
            raise ValueError(f"{field_name} is required for openai_remote models")
        return parsed

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None
