from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import DecodingParams
from app.schemas import EngineResult
from app.schemas import ImageContent
from app.schemas import ResponseMetrics
from app.schemas import ResponseRequest
from app.schemas import TextContent

from .common import LOGGER
from .common import BackendExecutionError
from .common import _exception_message
from .common import _merge_stop_strings
from .common import ResolvedDecoding


_REMOTE_IMAGE_TIMEOUT_S = 15.0
_REMOTE_IMAGE_MAX_BYTES = 16 * 1024 * 1024
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Return only the response."


def _ensure_blackwell_runtime_env() -> None:
    """Make vLLM load on Blackwell (SM 12.0) with a CUDA < 12.9 JIT toolchain.

    Two issues on RTX 50-series with the bundled CUDA 13 torch wheels but a
    system nvcc < 12.9:
    - flashinfer's JIT sampler builds for ``compute_120f``, which the older
      nvcc rejects. Disabling the flashinfer sampler falls back to the native
      PyTorch sampler.
    - flashinfer's JIT needs the ``ninja`` console script, which lives in the
      active venv's bin dir; that dir is not always on PATH under uv.

    Both are only applied when not already set, so a caller can override.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    venv_bin = str(Path(sys.executable).resolve().parent)
    path_value = os.environ.get("PATH", "")
    path_parts = path_value.split(os.pathsep) if path_value else []
    if venv_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([venv_bin, *path_parts])


@dataclass
class VllmModelRuntime:
    config: ModelSettings
    engine: object  # vllm.AsyncLLMEngine
    tokenizer: object
    loop: asyncio.AbstractEventLoop
    loop_thread: threading.Thread

    def shutdown(self) -> None:
        try:
            engine = self.engine
            if engine is not None:
                shutdown = getattr(engine, "shutdown", None)
                if callable(shutdown):
                    self._run_sync(shutdown() if asyncio.iscoroutinefunction(shutdown) else None)
        except Exception:
            LOGGER.exception("Error while shutting down vLLM engine for model %s", self.config.model_path)
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=5.0)
        except Exception:
            LOGGER.exception("Error while stopping vLLM event loop")

    def _run_sync(self, coro_or_none) -> None:
        if coro_or_none is None:
            return
        future = asyncio.run_coroutine_threadsafe(coro_or_none, self.loop)
        try:
            future.result(timeout=10.0)
        except Exception:
            LOGGER.exception("Error while awaiting vLLM engine shutdown coroutine")


class VllmEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, VllmModelRuntime] = {}
        self._load_errors: dict[str, str] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_settings)
            except Exception as exc:
                self._load_errors[model_name] = _exception_message(exc)
                LOGGER.exception(
                    "Failed to load vLLM model '%s' from %s; skipping model.",
                    model_name,
                    model_settings.model_path,
                )
        if not self._models:
            raise ValueError("no enabled models could be loaded")

    def complete(self, request: ResponseRequest) -> EngineResult:
        runtime = self._models.get(request.model)
        if runtime is None:
            raise ValueError(f"unknown model: {request.model!r}")

        decoding = self._resolve_decoding(request.decoding)
        stop_strings = _merge_stop_strings(runtime.config.prompt_format, decoding.stop)
        system_prompt = request.instructions or _DEFAULT_SYSTEM_PROMPT

        text_segments, images = self._extract_text_and_images(request)

        tokenize_started = time.perf_counter()
        prompt_text = self._render_prompt(
            runtime=runtime,
            system_prompt=system_prompt,
            user_text="".join(text_segments),
            has_images=bool(images),
        )
        tokenize_ms = max(0.0, (time.perf_counter() - tokenize_started) * 1000.0)

        engine_input: dict[str, Any] = {"prompt": prompt_text}
        if images:
            engine_input["multi_modal_data"] = {"image": images}

        sampling_params = self._build_sampling_params(decoding, stop_strings)

        generate_started = time.perf_counter()
        first_token_ms_ref: dict[str, float | None] = {"value": None}
        future = asyncio.run_coroutine_threadsafe(
            self._run_generation(
                runtime=runtime,
                engine_input=engine_input,
                sampling_params=sampling_params,
                first_token_ms_ref=first_token_ms_ref,
                generate_started=generate_started,
            ),
            runtime.loop,
        )
        try:
            output_text, output_tokens, prompt_token_count = future.result()
        except BackendExecutionError:
            raise
        except Exception as exc:
            message = _exception_message(exc)
            raise BackendExecutionError(
                code="vllm_generation_failed",
                status_code=500,
                message=message,
            ) from exc

        generate_total_ms = max(0.0, (time.perf_counter() - generate_started) * 1000.0)
        gpu_decode_after_first_token_ms = None
        first_token_ms = first_token_ms_ref["value"]
        if first_token_ms is not None:
            gpu_decode_after_first_token_ms = max(0.0, generate_total_ms - first_token_ms)
        tokens_per_second = None
        if generate_total_ms > 0.0 and output_tokens > 0:
            tokens_per_second = output_tokens / (generate_total_ms / 1000.0)

        return EngineResult(
            text=output_text.strip(),
            metrics=ResponseMetrics(
                engine_tokenize_ms=tokenize_ms,
                gpu_time_to_first_token_ms=first_token_ms,
                gpu_generate_total_ms=generate_total_ms,
                gpu_decode_after_first_token_ms=gpu_decode_after_first_token_ms,
                engine_prompt_tokens=prompt_token_count,
                engine_output_tokens=output_tokens,
                engine_tokens_per_second=tokens_per_second,
            ),
        )

    def close(self) -> None:
        for runtime in self._models.values():
            try:
                runtime.shutdown()
            except Exception:
                LOGGER.exception("Error while shutting down vLLM runtime")
        self._models.clear()

    def _build_runtime(self, settings: ModelSettings) -> VllmModelRuntime:
        _ensure_blackwell_runtime_env()
        from vllm import AsyncEngineArgs
        from vllm import AsyncLLMEngine

        model_ref = (settings.vllm_model or settings.model_path or "").strip()
        if not model_ref:
            raise ValueError("vllm backend requires model_path or vllm_model to be set")

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=self._run_event_loop,
            args=(loop,),
            name=f"vllm-loop-{model_ref.replace('/', '_')[:50]}",
            daemon=True,
        )
        loop_thread.start()

        engine_args_kwargs: dict[str, Any] = {
            "model": model_ref,
            "dtype": settings.vllm_dtype,
            "tensor_parallel_size": settings.vllm_tensor_parallel_size,
            "trust_remote_code": settings.vllm_trust_remote_code,
            "enforce_eager": settings.vllm_enforce_eager,
        }
        # Prefer kv_cache_memory_bytes for a machine-independent footprint: it
        # sets an absolute KV cache size and ignores gpu_memory_utilization, so
        # the same value behaves identically across GPUs of different sizes.
        if settings.vllm_kv_cache_memory_bytes is not None:
            engine_args_kwargs["kv_cache_memory_bytes"] = settings.vllm_kv_cache_memory_bytes
        if settings.vllm_gpu_memory_utilization is not None:
            engine_args_kwargs["gpu_memory_utilization"] = settings.vllm_gpu_memory_utilization
        if settings.vllm_kv_cache_dtype and settings.vllm_kv_cache_dtype != "auto":
            engine_args_kwargs["kv_cache_dtype"] = settings.vllm_kv_cache_dtype
        if settings.vllm_max_model_len is not None:
            engine_args_kwargs["max_model_len"] = settings.vllm_max_model_len
        if settings.vllm_limit_mm_per_prompt:
            engine_args_kwargs["limit_mm_per_prompt"] = dict(settings.vllm_limit_mm_per_prompt)
        if settings.vllm_mm_processor_kwargs:
            engine_args_kwargs["mm_processor_kwargs"] = dict(settings.vllm_mm_processor_kwargs)
        if settings.vllm_speculative_method:
            engine_args_kwargs["speculative_config"] = {
                "method": settings.vllm_speculative_method,
                "model": settings.vllm_speculative_model,
                "num_speculative_tokens": settings.vllm_num_speculative_tokens,
            }

        engine_args = AsyncEngineArgs(**engine_args_kwargs)

        async def _create_engine() -> object:
            return AsyncLLMEngine.from_engine_args(engine_args)

        engine = asyncio.run_coroutine_threadsafe(_create_engine(), loop).result()

        async def _get_tokenizer() -> object:
            result = engine.get_tokenizer()
            if asyncio.iscoroutine(result):
                return await result
            return result

        tokenizer = asyncio.run_coroutine_threadsafe(_get_tokenizer(), loop).result()

        return VllmModelRuntime(
            config=settings,
            engine=engine,
            tokenizer=tokenizer,
            loop=loop,
            loop_thread=loop_thread,
        )

    @staticmethod
    def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass

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

    def _build_sampling_params(self, decoding: ResolvedDecoding, stop_strings: list[str]):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k if decoding.top_k > 0 else -1,
            repetition_penalty=decoding.repetition_penalty,
            max_tokens=decoding.max_tokens,
            stop=stop_strings if stop_strings else None,
        )

    def _extract_text_and_images(self, request: ResponseRequest) -> tuple[list[str], list[Any]]:
        if isinstance(request.input, str):
            return [request.input], []
        text_segments: list[str] = []
        images: list[Any] = []
        for item in request.input:
            if isinstance(item, TextContent):
                text_segments.append(item.text)
            elif isinstance(item, ImageContent):
                images.append(self._load_image(item))
        return text_segments, images

    def _load_image(self, item: ImageContent) -> Any:
        from PIL import Image

        url = item.image_url.url.strip()
        if not url:
            raise BackendExecutionError(
                code="image_url_empty",
                status_code=400,
                message="image_url.url must not be empty",
            )
        if url.startswith("data:"):
            data = self._decode_data_url(url)
        elif url.startswith("file://"):
            parsed = urlparse(url)
            with open(parsed.path, "rb") as handle:
                data = handle.read()
        elif url.startswith("http://") or url.startswith("https://"):
            data = self._fetch_remote_bytes(url)
        else:
            raise BackendExecutionError(
                code="image_url_scheme_unsupported",
                status_code=400,
                message="image_url.url must be a data: URL, file:// URL, or http(s):// URL",
            )
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            raise BackendExecutionError(
                code="image_decode_failed",
                status_code=400,
                message=f"failed to decode image: {_exception_message(exc)}",
            ) from exc
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        return image

    @staticmethod
    def _decode_data_url(url: str) -> bytes:
        try:
            header, payload = url.split(",", 1)
        except ValueError as exc:
            raise BackendExecutionError(
                code="image_data_url_invalid",
                status_code=400,
                message="data: URL must contain a comma-separated payload",
            ) from exc
        if ";base64" not in header:
            raise BackendExecutionError(
                code="image_data_url_invalid",
                status_code=400,
                message="data: URL must be base64-encoded",
            )
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise BackendExecutionError(
                code="image_data_url_invalid",
                status_code=400,
                message=f"failed to decode base64 payload: {_exception_message(exc)}",
            ) from exc

    @staticmethod
    def _fetch_remote_bytes(url: str) -> bytes:
        request = UrlRequest(url, headers={"User-Agent": "llm-pool/vllm-backend"})
        with urlopen(request, timeout=_REMOTE_IMAGE_TIMEOUT_S) as response:
            data = response.read(_REMOTE_IMAGE_MAX_BYTES + 1)
        if len(data) > _REMOTE_IMAGE_MAX_BYTES:
            raise BackendExecutionError(
                code="image_too_large",
                status_code=400,
                message=f"remote image exceeds {_REMOTE_IMAGE_MAX_BYTES} bytes",
            )
        return data

    def _render_prompt(
        self,
        *,
        runtime: VllmModelRuntime,
        system_prompt: str,
        user_text: str,
        has_images: bool,
    ) -> str:
        user_content: Any
        if has_images:
            user_content = [{"type": "image"}, {"type": "text", "text": user_text}]
        else:
            user_content = user_text
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            return runtime.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as exc:
            raise BackendExecutionError(
                code="chat_template_render_failed",
                status_code=500,
                message=f"failed to render chat template: {_exception_message(exc)}",
            ) from exc

    async def _run_generation(
        self,
        *,
        runtime: VllmModelRuntime,
        engine_input: dict[str, Any],
        sampling_params: Any,
        first_token_ms_ref: dict[str, float | None],
        generate_started: float,
    ) -> tuple[str, int, int]:
        request_id = f"vllm-{uuid.uuid4().hex}"
        output_text = ""
        output_tokens = 0
        prompt_token_count = 0
        async for request_output in runtime.engine.generate(
            engine_input,
            sampling_params,
            request_id=request_id,
        ):
            if first_token_ms_ref["value"] is None and request_output.outputs:
                first_token_ms_ref["value"] = max(
                    0.0,
                    (time.perf_counter() - generate_started) * 1000.0,
                )
            if request_output.prompt_token_ids is not None:
                prompt_token_count = len(request_output.prompt_token_ids)
            if request_output.outputs:
                completion = request_output.outputs[0]
                output_text = completion.text
                output_tokens = len(completion.token_ids)
        return output_text, output_tokens, prompt_token_count


