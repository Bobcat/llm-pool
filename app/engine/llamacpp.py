from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import threading
import time

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import DecodingParams
from app.schemas import EngineResult
from app.schemas import ResponseRequest
from app.schemas import ResponseMetrics

from .common import LOGGER
from .common import _exception_message
from .common import _merge_stop_strings
from .common import _require_text_input
from .common import _resolve_gguf_cache_type_constant
from .common import ResolvedDecoding

_LLAMA_CPP_FLASH_ATTN_MODE_LOCK = threading.Lock()


@dataclass
class LlamaCppModelRuntime:
    config: ModelSettings
    llm: object
    generation_lock: threading.Lock = field(default_factory=threading.Lock)


class LlamaCppEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, LlamaCppModelRuntime] = {}
        self._load_errors: dict[str, str] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_settings)
            except Exception as exc:
                self._load_errors[model_name] = _exception_message(exc)
                LOGGER.exception(
                    "Failed to load model '%s' from %s; skipping model.",
                    model_name,
                    model_settings.model_path,
                )
        if not self._models:
            raise ValueError("no enabled models could be loaded")

    def complete(self, request: ResponseRequest) -> EngineResult:
        runtime = self._models.get(request.model)
        if runtime is None:
            raise ValueError(f"unknown model: {request.model!r}")

        user_text = _require_text_input(request)
        system_prompt = request.instructions or "You are a helpful assistant. Return only the response."
        decoding = self._resolve_decoding(request.decoding)
        prompt_format = runtime.config.prompt_format
        stop_strings = self._resolve_stop_strings(prompt_format, decoding.stop)
        if request.decoding.beam_size is not None:
            LOGGER.info(
                "Ignoring beam_size=%s for GGUF model '%s'.",
                request.decoding.beam_size,
                request.model,
            )
        if prompt_format == "gemma4_template":
            return self._complete_with_native_chat_template(
                runtime=runtime,
                user_text=user_text,
                system_prompt=system_prompt,
                decoding=decoding,
                stop_strings=stop_strings,
            )
        if prompt_format == "translategemma_template":
            return self._complete_with_translategemma_template(
                runtime=runtime,
                user_text=user_text,
                source_lang_code=request.source_lang_code,
                target_lang_code=request.target_lang_code,
                decoding=decoding,
                stop_strings=stop_strings,
            )

        tokenize_started = time.perf_counter()
        prompt_text = self._render_prompt(
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            user_text=user_text,
            enable_thinking=runtime.config.enable_thinking,
        )
        prompt_tokens = runtime.llm.tokenize(
            prompt_text.encode("utf-8"),
            add_bos=False,
            special=True,
        )
        prompt_token_count = len(prompt_tokens)
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0

        generate_started = time.perf_counter()
        first_token_ms = None
        output_token_ids: list[int] = []
        text = ""
        with runtime.generation_lock:
            for token_id in runtime.llm.generate(
                prompt_tokens,
                top_k=decoding.top_k,
                top_p=decoding.top_p,
                temp=decoding.temperature,
                repeat_penalty=decoding.repetition_penalty,
            ):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - generate_started) * 1000.0
                output_token_ids.append(token_id)
                text = self._decode_output(runtime.llm, output_token_ids)
                if token_id == runtime.llm.token_eos():
                    break
                if len(output_token_ids) >= decoding.max_tokens:
                    break
                stripped_text = self._strip_stop_suffix(text, stop_strings)
                if stripped_text is not None:
                    text = stripped_text
                    break

        generate_total_ms = (time.perf_counter() - generate_started) * 1000.0
        if text == "":
            text = self._decode_output(runtime.llm, output_token_ids)
        stripped_text = self._strip_stop_suffix(text, stop_strings)
        if stripped_text is not None:
            text = stripped_text
        output_tokens = len(output_token_ids)
        text = text.strip()

        gpu_decode_after_first_token_ms = None
        if first_token_ms is not None:
            gpu_decode_after_first_token_ms = max(0.0, generate_total_ms - first_token_ms)
        engine_tokens_per_second = None
        if generate_total_ms > 0.0:
            engine_tokens_per_second = output_tokens / (generate_total_ms / 1000.0)
        return EngineResult(
            text=text,
            metrics=ResponseMetrics(
                engine_tokenize_ms=tokenize_ms,
                gpu_time_to_first_token_ms=first_token_ms,
                gpu_generate_total_ms=generate_total_ms,
                gpu_decode_after_first_token_ms=gpu_decode_after_first_token_ms,
                engine_prompt_tokens=prompt_token_count,
                engine_output_tokens=output_tokens,
                engine_tokens_per_second=engine_tokens_per_second,
            ),
        )

    def _complete_with_native_chat_template(
        self,
        *,
        runtime: LlamaCppModelRuntime,
        user_text: str,
        system_prompt: str,
        decoding: ResolvedDecoding,
        stop_strings: list[str],
    ) -> EngineResult:
        generate_started = time.perf_counter()
        with runtime.generation_lock:
            response = runtime.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
                max_tokens=decoding.max_tokens,
                repeat_penalty=decoding.repetition_penalty,
                stop=stop_strings,
            )
        generate_total_ms = (time.perf_counter() - generate_started) * 1000.0
        usage = response.get("usage") or {}
        prompt_token_count = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        engine_tokens_per_second = None
        if output_tokens is not None and generate_total_ms > 0.0:
            engine_tokens_per_second = output_tokens / (generate_total_ms / 1000.0)
        text = response["choices"][0]["message"]["content"].strip()
        return EngineResult(
            text=text,
            metrics=ResponseMetrics(
                gpu_generate_total_ms=generate_total_ms,
                engine_prompt_tokens=prompt_token_count,
                engine_output_tokens=output_tokens,
                engine_tokens_per_second=engine_tokens_per_second,
            ),
        )

    def _complete_with_translategemma_template(
        self,
        *,
        runtime: LlamaCppModelRuntime,
        user_text: str,
        source_lang_code: str | None,
        target_lang_code: str | None,
        decoding: ResolvedDecoding,
        stop_strings: list[str],
    ) -> EngineResult:
        source_lang_code = self._resolve_translation_lang_code(
            source_lang_code,
            "source_lang_code",
        )
        target_lang_code = self._resolve_translation_lang_code(
            target_lang_code,
            "target_lang_code",
        )

        generate_started = time.perf_counter()
        with runtime.generation_lock:
            response = runtime.llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "source_lang_code": source_lang_code,
                                "target_lang_code": target_lang_code,
                                "text": user_text,
                            }
                        ],
                    },
                ],
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
                max_tokens=decoding.max_tokens,
                repeat_penalty=decoding.repetition_penalty,
                stop=stop_strings,
            )
        generate_total_ms = (time.perf_counter() - generate_started) * 1000.0
        usage = response.get("usage") or {}
        prompt_token_count = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        engine_tokens_per_second = None
        if output_tokens is not None and generate_total_ms > 0.0:
            engine_tokens_per_second = output_tokens / (generate_total_ms / 1000.0)
        text = response["choices"][0]["message"]["content"].strip()
        return EngineResult(
            text=text,
            metrics=ResponseMetrics(
                gpu_generate_total_ms=generate_total_ms,
                engine_prompt_tokens=prompt_token_count,
                engine_output_tokens=output_tokens,
                engine_tokens_per_second=engine_tokens_per_second,
            ),
        )

    def _resolve_translation_lang_code(self, value: str | None, field_name: str) -> str:
        if value is None:
            raise ValueError(f"translategemma_template requires request.{field_name}")
        normalized = value.strip()
        if normalized == "":
            raise ValueError(f"translategemma_template requires non-empty request.{field_name}")
        return normalized

    def _build_runtime(self, settings: ModelSettings) -> LlamaCppModelRuntime:
        try:
            import llama_cpp as llama_cpp_module
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("llama-cpp-python is required for the GGUF engine") from exc
        try:
            import llama_cpp.llama_cpp as llama_cpp_low_level
        except ImportError:
            llama_cpp_low_level = getattr(llama_cpp_module, "llama_cpp", llama_cpp_module)

        llm = self._build_llama(
            llama_constructor=Llama,
            llama_cpp_module=llama_cpp_module,
            llama_cpp_low_level=llama_cpp_low_level,
            model_path=settings.model_path,
            n_gpu_layers=settings.gguf_n_gpu_layers,
            n_ctx=settings.gguf_n_ctx,
            flash_attn_mode=settings.gguf_flash_attn,
            type_k=self._resolve_cache_type_constant(settings.gguf_type_k, llama_cpp_module=llama_cpp_module),
            type_v=self._resolve_cache_type_constant(settings.gguf_type_v, llama_cpp_module=llama_cpp_module),
        )
        return LlamaCppModelRuntime(config=settings, llm=llm)

    def _build_llama(
        self,
        *,
        llama_constructor,
        llama_cpp_module: object,
        llama_cpp_low_level: object,
        model_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        flash_attn_mode: str,
        type_k,
        type_v,
    ):
        kwargs = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "type_k": type_k,
            "type_v": type_v,
            "verbose": False,
        }
        if flash_attn_mode == "on":
            return llama_constructor(flash_attn=True, **kwargs)
        if flash_attn_mode == "off":
            return llama_constructor(flash_attn=False, **kwargs)
        if flash_attn_mode == "auto":
            return self._build_llama_with_auto_flash_attn(
                llama_constructor=llama_constructor,
                llama_cpp_module=llama_cpp_module,
                llama_cpp_low_level=llama_cpp_low_level,
                **kwargs,
            )
        raise ValueError(f"unsupported gguf_flash_attn mode: {flash_attn_mode!r}")

    def _build_llama_with_auto_flash_attn(
        self,
        *,
        llama_constructor,
        llama_cpp_module: object,
        llama_cpp_low_level: object,
        **kwargs,
    ):
        auto_constant = getattr(llama_cpp_low_level, "LLAMA_FLASH_ATTN_TYPE_AUTO", None)
        if auto_constant is None:
            raise RuntimeError("llama-cpp-python does not expose LLAMA_FLASH_ATTN_TYPE_AUTO")

        namespaces: list[object] = []
        for namespace in (llama_cpp_low_level, getattr(llama_cpp_module, "llama_cpp", None), llama_cpp_module):
            if namespace is not None and hasattr(namespace, "LLAMA_FLASH_ATTN_TYPE_DISABLED"):
                if namespace not in namespaces:
                    namespaces.append(namespace)
        if not namespaces:
            raise RuntimeError("llama-cpp-python does not expose LLAMA_FLASH_ATTN_TYPE_DISABLED")

        originals = {
            id(namespace): getattr(namespace, "LLAMA_FLASH_ATTN_TYPE_DISABLED")
            for namespace in namespaces
        }
        # The installed llama-cpp-python binding still exposes a bool flash_attn argument,
        # so set the underlying enum to AUTO for this constructor call only.
        with _LLAMA_CPP_FLASH_ATTN_MODE_LOCK:
            try:
                for namespace in namespaces:
                    setattr(namespace, "LLAMA_FLASH_ATTN_TYPE_DISABLED", auto_constant)
                return llama_constructor(flash_attn=False, **kwargs)
            finally:
                for namespace in namespaces:
                    setattr(namespace, "LLAMA_FLASH_ATTN_TYPE_DISABLED", originals[id(namespace)])

    def _resolve_cache_type_constant(self, cache_type_name: str | None, *, llama_cpp_module: object):
        if cache_type_name is None:
            return None
        return _resolve_gguf_cache_type_constant(cache_type_name, llama_cpp_module=llama_cpp_module)

    def _render_prompt(
        self,
        *,
        prompt_format: str,
        system_prompt: str,
        user_text: str,
        enable_thinking: bool | None,
    ) -> str:
        if prompt_format == "qwen3_template":
            qwen_user_text = user_text
            assistant_prefix = "<|im_start|>assistant\n"
            if enable_thinking is not True:
                if not qwen_user_text.lstrip().startswith("/no_think"):
                    qwen_user_text = f"/no_think\n{qwen_user_text}"
                assistant_prefix += "<think>\n\n</think>\n\n"
            return (
                "<|im_start|>system\n"
                f"{system_prompt}<|im_end|>\n"
                "<|im_start|>user\n"
                f"{qwen_user_text}<|im_end|>\n"
                f"{assistant_prefix}"
            )
        if prompt_format == "mistral_template":
            merged_user_content = f"{system_prompt}\n\n{user_text}"
            return f"<s>[INST] {merged_user_content} [/INST]"
        if prompt_format == "gemma4_template":
            return (
                "<start_of_turn>system\n"
                f"{system_prompt}<end_of_turn>\n"
                "<start_of_turn>user\n"
                f"{user_text}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        return (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _decode_output(self, llm: object, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        text_bytes = llm.detokenize(token_ids)
        return text_bytes.decode("utf-8", errors="replace")

    def _resolve_stop_strings(self, prompt_format: str, extra_stop_tokens: list[str]) -> list[str]:
        stop_strings = _merge_stop_strings(prompt_format, extra_stop_tokens)
        if prompt_format == "gemma4_template" and "<end_of_turn>" not in stop_strings:
            stop_strings.append("<end_of_turn>")
        return stop_strings

    def _strip_stop_suffix(self, text: str, stop_strings: list[str]) -> str | None:
        for stop_str in stop_strings:
            if stop_str != "" and text.endswith(stop_str):
                return text[: -len(stop_str)]
        return None

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
