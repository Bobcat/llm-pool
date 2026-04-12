from __future__ import annotations

import gc
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
import logging
import threading
import time

from app.config import AppSettings
from app.config import DecodingDefaults
from app.config import ModelSettings
from app.schemas import DecodingParams
from app.schemas import EngineResult
from app.schemas import ResponseRequest
from app.schemas import ResponseMetrics

LOGGER = logging.getLogger("llm_pool.engine")


def _native_stop_strings(prompt_format: str) -> list[str]:
    if prompt_format in {"generic", "qwen3_template"}:
        return ["<|im_end|>"]
    if prompt_format == "gemma4_template":
        return ["<turn|>"]
    return []


def _merge_stop_strings(prompt_format: str, extra_stop_tokens: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for token in [*_native_stop_strings(prompt_format), *extra_stop_tokens]:
        if token == "" or token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message != "":
        return message
    return exc.__class__.__name__


class UnknownModelError(LookupError):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.model_name = model_name


class ModelStateError(RuntimeError):
    def __init__(self, model_name: str, code: str) -> None:
        super().__init__(model_name)
        self.model_name = model_name
        self.code = code


class StubEngine:
    """Temporary engine used to validate the API contract before model integration."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._configured_models = dict(settings.engine.models) if settings is not None else {}
        self._models = {
            model_name: object()
            for model_name, model_settings in self._configured_models.items()
            if model_settings.enabled
        }

    def complete(self, request: ResponseRequest) -> EngineResult:
        if request.model not in self._configured_models:
            raise UnknownModelError(request.model)
        if request.model not in self._models:
            raise ModelStateError(request.model, "model_not_loaded")
        text = request.input
        if request.instructions:
            text = f"[instructions={request.instructions}] {text}"
        return EngineResult(text=text)

    def admin_models_payload(self, settings: AppSettings) -> dict[str, object]:
        models: list[dict[str, object]] = []
        for model_name, model_settings in self._configured_models.items():
            is_loaded = model_name in self._models
            models.append(
                {
                    "name": model_name,
                    "resolved_backend": settings.engine.backend,
                    "configured_enabled": model_settings.enabled,
                    "runtime_state": "loaded" if is_loaded else "unloaded",
                    "is_loaded": is_loaded,
                    "inflight_requests": 0,
                    "last_error": None,
                    "definition": asdict(model_settings),
                }
            )
        return {"models": models}

    def load_model(self, model_name: str, settings: AppSettings) -> dict[str, object]:
        model_settings = self._configured_models.get(model_name)
        if model_settings is None:
            raise UnknownModelError(model_name)
        self._models.setdefault(model_name, object())
        return {
            "name": model_name,
            "resolved_backend": settings.engine.backend,
            "configured_enabled": model_settings.enabled,
            "runtime_state": "loaded",
            "is_loaded": True,
            "inflight_requests": 0,
            "last_error": None,
            "definition": asdict(model_settings),
        }

    def unload_model(self, model_name: str, settings: AppSettings) -> dict[str, object]:
        model_settings = self._configured_models.get(model_name)
        if model_settings is None:
            raise UnknownModelError(model_name)
        self._models.pop(model_name, None)
        return {
            "name": model_name,
            "resolved_backend": settings.engine.backend,
            "configured_enabled": model_settings.enabled,
            "runtime_state": "unloaded",
            "is_loaded": False,
            "inflight_requests": 0,
            "last_error": None,
            "definition": asdict(model_settings),
        }


@dataclass
class Ct2ModelRuntime:
    config: ModelSettings
    generator: object
    tokenizer: object
    prompt_token_cache: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ExLlamaV3ModelRuntime:
    config: ModelSettings
    model: object
    cache: object
    tokenizer: object
    generator: object
    job_class: object
    sampler_class: object
    generation_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class LlamaCppModelRuntime:
    config: ModelSettings
    llm: object
    generation_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class ModelRuntimeState:
    resolved_backend: str
    configured_enabled: bool
    lifecycle: str = "unloaded"
    inflight_requests: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class ResolvedDecoding:
    beam_size: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    max_tokens: int
    stop: list[str]


class Ct2Engine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, Ct2ModelRuntime] = {}
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

        system_prompt = request.instructions or "You are a helpful assistant. Return only the response."
        decoding = self._resolve_decoding(request.decoding)
        stop_tokens = _merge_stop_strings(runtime.config.prompt_format, decoding.stop)
        tokenize_started = time.perf_counter()
        prompt_token_count = 0
        if runtime.config.prompt_format == "qwen3_template":
            suppress_sequences = None
            if runtime.config.enable_thinking is not True:
                think_open_tokens = self._tokenize(runtime.tokenizer, "<think>", add_special_tokens=False)
                think_close_tokens = self._tokenize(runtime.tokenizer, "</think>", add_special_tokens=False)
                suppress_sequences = [tokens for tokens in (think_open_tokens, think_close_tokens) if tokens]
            prompt_tokens = self._render_qwen3_prompt_tokens(
                runtime.tokenizer,
                system_prompt=system_prompt,
                user_text=request.input,
                enable_thinking=runtime.config.enable_thinking,
            )
            prompt_token_count = len(prompt_tokens)
            tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0
            generate_started = time.perf_counter()
            callback, first_token_ms_ref = self._build_first_token_callback(generate_started)
            end_token = self._resolve_end_token(runtime.tokenizer, stop_tokens)
            generate_kwargs = {
                "include_prompt_in_result": False,
                "beam_size": decoding.beam_size,
                "max_length": decoding.max_tokens,
                "sampling_topk": decoding.top_k,
                "sampling_topp": decoding.top_p,
                "sampling_temperature": decoding.temperature,
                "repetition_penalty": decoding.repetition_penalty,
                "suppress_sequences": suppress_sequences,
                "callback": callback if decoding.beam_size == 1 else None,
            }
            if end_token is not None:
                generate_kwargs["end_token"] = end_token
            result = runtime.generator.generate_batch(  # type: ignore[call-arg]
                [prompt_tokens],
                **generate_kwargs,
            )
        elif runtime.config.prompt_format == "mistral_template":
            prompt_tokens = self._render_mistral_prompt_tokens(
                runtime.tokenizer,
                system_prompt=system_prompt,
                user_text=request.input,
            )
            prompt_token_count = len(prompt_tokens)
            tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0
            generate_started = time.perf_counter()
            callback, first_token_ms_ref = self._build_first_token_callback(generate_started)
            end_token = self._resolve_end_token(runtime.tokenizer, stop_tokens)
            generate_kwargs = {
                "include_prompt_in_result": False,
                "beam_size": decoding.beam_size,
                "max_length": decoding.max_tokens,
                "sampling_topk": decoding.top_k,
                "sampling_topp": decoding.top_p,
                "sampling_temperature": decoding.temperature,
                "repetition_penalty": decoding.repetition_penalty,
                "callback": callback if decoding.beam_size == 1 else None,
            }
            if end_token is not None:
                generate_kwargs["end_token"] = end_token
            result = runtime.generator.generate_batch(  # type: ignore[call-arg]
                [prompt_tokens],
                **generate_kwargs,
            )
        else:
            user_text = f"{request.input}<|im_end|>\n<|im_start|>assistant\n"
            request_tokens = self._tokenize(runtime.tokenizer, user_text, add_special_tokens=False)
            static_prompt_tokens = self._get_static_prompt_tokens(runtime, system_prompt)
            prompt_token_count = len(static_prompt_tokens) + len(request_tokens)
            tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0
            generate_started = time.perf_counter()
            callback, first_token_ms_ref = self._build_first_token_callback(generate_started)
            end_token = self._resolve_end_token(runtime.tokenizer, stop_tokens)
            generate_kwargs = {
                "static_prompt": static_prompt_tokens,
                "cache_static_prompt": True,
                "include_prompt_in_result": False,
                "beam_size": decoding.beam_size,
                "max_length": decoding.max_tokens,
                "sampling_topk": decoding.top_k,
                "sampling_topp": decoding.top_p,
                "sampling_temperature": decoding.temperature,
                "repetition_penalty": decoding.repetition_penalty,
                "callback": callback if decoding.beam_size == 1 else None,
            }
            if end_token is not None:
                generate_kwargs["end_token"] = end_token
            result = runtime.generator.generate_batch(  # type: ignore[call-arg]
                [request_tokens],
                **generate_kwargs,
            )
        generate_total_ms = (time.perf_counter() - generate_started) * 1000.0
        first_token_ms = first_token_ms_ref["value"]
        output_tokens = 0
        if not result or not result[0].sequences:
            text = ""
        else:
            output_tokens = len(result[0].sequences[0])
            text = self._decode(runtime.tokenizer, result[0].sequences[0]).strip()
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

    def _build_runtime(self, settings: ModelSettings) -> Ct2ModelRuntime:
        try:
            import ctranslate2
            from transformers import AutoTokenizer
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("ctranslate2 and transformers are required for the CT2 engine") from exc

        model_path = Path(settings.model_path)
        generator = ctranslate2.Generator(
            str(model_path),
            device=settings.device,
            compute_type=settings.compute_type,
        )
        if settings.prompt_format in {"qwen3_template", "mistral_template"}:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        else:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(model_path / "tokenizer.json"),
                bos_token="<s>",
                eos_token="<|im_end|>",
                unk_token="<unk>",
            )
        return Ct2ModelRuntime(config=settings, generator=generator, tokenizer=tokenizer)

    def _get_static_prompt_tokens(self, runtime: Ct2ModelRuntime, system_prompt: str) -> list[str]:
        cached = runtime.prompt_token_cache.get(system_prompt)
        if cached is not None:
            return cached
        prompt_text = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        tokens = self._tokenize(runtime.tokenizer, prompt_text, add_special_tokens=True)
        runtime.prompt_token_cache[system_prompt] = tokens
        return tokens

    def _tokenize(self, tokenizer: object, text: str, *, add_special_tokens: bool) -> list[str]:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)
        return tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    def _render_qwen3_prompt_tokens(
        self,
        tokenizer: object,
        *,
        system_prompt: str,
        user_text: str,
        enable_thinking: bool | None,
    ) -> list[str]:
        qwen_user_text = user_text
        assistant_prefix = "<|im_start|>assistant\n"
        if enable_thinking is not True:
            if not qwen_user_text.lstrip().startswith("/no_think"):
                qwen_user_text = f"/no_think\n{qwen_user_text}"
            assistant_prefix += "<think>\n\n</think>\n\n"
        prompt_text = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{qwen_user_text}<|im_end|>\n"
            f"{assistant_prefix}"
        )
        return self._tokenize(tokenizer, prompt_text, add_special_tokens=False)

    def _render_mistral_prompt_tokens(self, tokenizer: object, *, system_prompt: str, user_text: str) -> list[str]:
        # Mistral-7B-Instruct-v0.3 tokenizer template accepts user/assistant turns.
        merged_user_content = f"{system_prompt}\n\n{user_text}"
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_user_content}],
            tokenize=False,
            add_generation_prompt=True,
            return_tensors=None,
        )
        return self._tokenize(tokenizer, str(prompt_text), add_special_tokens=False)

    def _decode(self, tokenizer: object, tokens: list[str]) -> str:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def _resolve_end_token(self, tokenizer: object, stop_tokens: list[str]) -> str | None:
        for token in stop_tokens:
            if self._token_in_vocabulary(tokenizer, token):
                return token
        eos_token = getattr(tokenizer, "eos_token", None)
        if isinstance(eos_token, str) and self._token_in_vocabulary(tokenizer, eos_token):
            return eos_token
        return None

    def _token_in_vocabulary(self, tokenizer: object, token: str) -> bool:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            try:
                vocab = get_vocab()
            except Exception:
                vocab = None
            if isinstance(vocab, dict):
                return token in vocab
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int):
            return False
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(unk_token_id, int):
            return token_id != unk_token_id
        return token_id >= 0

    def _build_first_token_callback(self, started: float):
        first_token_ms_ref: dict[str, float | None] = {"value": None}

        def callback(_step_result) -> bool:
            if first_token_ms_ref["value"] is None:
                first_token_ms_ref["value"] = (time.perf_counter() - started) * 1000.0
            return False

        return callback, first_token_ms_ref

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


class ExLlamaV3Engine:
    def __init__(self, settings: AppSettings) -> None:
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, ExLlamaV3ModelRuntime] = {}
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

        system_prompt = request.instructions or "You are a helpful assistant. Return only the response."
        decoding = self._resolve_decoding(request.decoding)
        prompt_format = runtime.config.prompt_format
        stop_tokens = _merge_stop_strings(prompt_format, decoding.stop)
        if request.decoding.beam_size is not None:
            LOGGER.info(
                "Ignoring beam_size=%s for ExLlamaV3 model '%s'.",
                request.decoding.beam_size,
                request.model,
            )

        tokenize_started = time.perf_counter()
        prompt_ids = self._render_prompt_ids(
            runtime.tokenizer,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            user_text=request.input,
            enable_thinking=runtime.config.enable_thinking,
        )
        prompt_ids = prompt_ids.to(device="cpu")
        prompt_token_count = int(prompt_ids.shape[-1])
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0

        sampler = runtime.sampler_class(
            rep_p=decoding.repetition_penalty,
            top_k=decoding.top_k,
            top_p=decoding.top_p,
            temperature=decoding.temperature,
            min_p=0.0,
        )
        stop_conditions = self._resolve_stop_conditions(runtime.tokenizer, stop_tokens)
        job_kwargs: dict[str, object] = {
            "input_ids": prompt_ids,
            "max_new_tokens": decoding.max_tokens,
            "sampler": sampler,
            "stop_conditions": stop_conditions,
        }
        if runtime.config.exllama_max_rq_tokens is not None:
            job_kwargs["max_rq_tokens"] = runtime.config.exllama_max_rq_tokens
        job = runtime.job_class(**job_kwargs)

        generate_started = time.perf_counter()
        eos_event: dict[str, object] | None = None
        text_chunks: list[str] = []
        with runtime.generation_lock:
            runtime.generator.enqueue(job)
            try:
                while runtime.generator.num_remaining_jobs() > 0:
                    events = runtime.generator.iterate()
                    for event in events:
                        if event.get("job") is not job:
                            continue
                        if event.get("stage") != "streaming":
                            continue
                        text_chunk = event.get("text")
                        if isinstance(text_chunk, str):
                            text_chunks.append(text_chunk)
                        if event.get("eos"):
                            eos_event = event
            finally:
                if runtime.generator.num_remaining_jobs() > 0:
                    runtime.generator.clear_queue()

        generate_total_ms = (time.perf_counter() - generate_started) * 1000.0
        first_token_ms = None
        gpu_decode_after_first_token_ms = None
        output_tokens = 0
        text = "".join(text_chunks)
        if eos_event is not None:
            full_completion = eos_event.get("full_completion")
            if isinstance(full_completion, str):
                text = full_completion
            new_tokens = eos_event.get("new_tokens")
            if isinstance(new_tokens, int):
                output_tokens = new_tokens
            time_prefill = eos_event.get("time_prefill")
            time_generate = eos_event.get("time_generate")
            if isinstance(time_prefill, (float, int)):
                first_token_ms = max(0.0, float(time_prefill) * 1000.0)
            if isinstance(time_generate, (float, int)):
                gpu_decode_after_first_token_ms = max(0.0, float(time_generate) * 1000.0)
            if first_token_ms is not None and gpu_decode_after_first_token_ms is not None:
                generate_total_ms = first_token_ms + gpu_decode_after_first_token_ms
        text = text.strip()

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

    def _build_runtime(self, settings: ModelSettings) -> ExLlamaV3ModelRuntime:
        try:
            from exllamav3 import Cache
            from exllamav3 import CacheLayer_fp16
            from exllamav3 import CacheLayer_quant
            from exllamav3 import ComboSampler
            from exllamav3 import Config
            from exllamav3 import Generator
            from exllamav3 import Job
            from exllamav3 import Model
            from exllamav3 import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("exllamav3 is required for the ExLlamaV3 engine") from exc

        model_path = Path(settings.model_path)
        config = Config.from_directory(str(model_path))
        model = Model.from_config(config)

        cache_size = int(settings.exllama_cache_size)
        if cache_size <= 0 or cache_size % 256 != 0:
            raise ValueError("exllama_cache_size must be a positive multiple of 256")

        layer_type = CacheLayer_fp16
        cache_kwargs: dict[str, int] = {}
        if settings.exllama_cache_quant:
            k_bits, v_bits = self._parse_cache_quant(settings.exllama_cache_quant)
            layer_type = CacheLayer_quant
            cache_kwargs = {
                "k_bits": k_bits,
                "v_bits": v_bits,
            }
        cache = Cache(
            model,
            max_num_tokens=cache_size,
            layer_type=layer_type,
            **cache_kwargs,
        )

        load_kwargs: dict[str, object] = {"progressbar": False}
        if settings.exllama_tensor_parallel:
            load_kwargs["tensor_p"] = True
            load_kwargs["tp_backend"] = settings.exllama_tp_backend
        if settings.exllama_gpu_split:
            if settings.exllama_gpu_split != "auto":
                load_kwargs["use_per_device"] = self._parse_gpu_split(settings.exllama_gpu_split)
        elif not settings.exllama_tensor_parallel:
            device = settings.device.strip() if settings.device else "cuda"
            if device == "cuda":
                device = "cuda:0"
            load_kwargs["device"] = device
        model.load(**load_kwargs)

        tokenizer = Tokenizer.from_config(config)
        generator = Generator(
            model,
            cache,
            tokenizer,
            max_batch_size=settings.exllama_max_batch_size,
            max_chunk_size=settings.exllama_max_chunk_size,
            max_q_size=settings.exllama_max_q_size,
        )
        return ExLlamaV3ModelRuntime(
            config=settings,
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            generator=generator,
            job_class=Job,
            sampler_class=ComboSampler,
        )

    def _render_prompt_ids(
        self,
        tokenizer: object,
        *,
        prompt_format: str,
        system_prompt: str,
        user_text: str,
        enable_thinking: bool | None,
    ):
        if prompt_format == "qwen3_template":
            return tokenizer.hf_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                add_generation_prompt=True,
                enable_thinking=False if enable_thinking is None else enable_thinking,
            )
        if prompt_format == "gemma4_template":
            return tokenizer.hf_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                add_generation_prompt=True,
                enable_thinking=False if enable_thinking is None else enable_thinking,
            )
        if prompt_format == "mistral_template":
            merged_user_content = f"{system_prompt}\n\n{user_text}"
            return tokenizer.hf_chat_template(
                [{"role": "user", "content": merged_user_content}],
                add_generation_prompt=True,
            )
        prompt_text = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return tokenizer.encode(prompt_text, encode_special_tokens=True)

    def _resolve_stop_conditions(self, tokenizer: object, stop_tokens: list[str]) -> list[str | int]:
        stop_conditions: list[str | int] = []
        stop_conditions.extend(token for token in stop_tokens if token != "")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            stop_conditions.append(eos_token_id)
        eos_token_ids = getattr(tokenizer, "eos_token_id_list", None)
        if isinstance(eos_token_ids, list):
            for token_id in eos_token_ids:
                if isinstance(token_id, int):
                    stop_conditions.append(token_id)

        deduped: list[str | int] = []
        seen: set[tuple[type, str | int]] = set()
        for condition in stop_conditions:
            marker = (type(condition), condition)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(condition)
        return deduped

    def _parse_cache_quant(self, cache_quant: str) -> tuple[int, int]:
        split = [part.strip() for part in cache_quant.split(",") if part.strip() != ""]
        if len(split) == 1:
            bits = int(split[0])
            return bits, bits
        if len(split) == 2:
            return int(split[0]), int(split[1])
        raise ValueError("exllama_cache_quant must be '<bits>' or '<k_bits>,<v_bits>'")

    def _parse_gpu_split(self, gpu_split: str) -> list[float]:
        values = [part.strip() for part in gpu_split.split(",")]
        parsed = [float(value) for value in values if value != ""]
        if not parsed:
            raise ValueError("exllama_gpu_split must contain at least one numeric value")
        return parsed

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
                user_text=request.input,
                system_prompt=system_prompt,
                decoding=decoding,
                stop_strings=stop_strings,
            )

        tokenize_started = time.perf_counter()
        prompt_text = self._render_prompt(
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            user_text=request.input,
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

    def _build_runtime(self, settings: ModelSettings) -> LlamaCppModelRuntime:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("llama-cpp-python is required for the GGUF engine") from exc

        llm = Llama(
            model_path=settings.model_path,
            n_gpu_layers=settings.gguf_n_gpu_layers,
            n_ctx=settings.gguf_n_ctx,
            flash_attn=settings.gguf_flash_attn,
            verbose=False,
        )
        return LlamaCppModelRuntime(config=settings, llm=llm)

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


class ModelRouterEngine:
    def __init__(self, settings: AppSettings) -> None:
        self._configured_models = dict(settings.engine.models)
        self._models: dict[str, object] = {}
        self._model_engines: dict[str, object] = {}
        self._model_states: dict[str, ModelRuntimeState] = {}
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)
        if not self._configured_models:
            raise ValueError("no configured models")
        grouped_models: dict[str, dict[str, ModelSettings]] = {}
        for model_name, model_settings in settings.engine.models.items():
            backend = self._resolve_model_backend(settings.engine.backend, model_settings)
            self._model_states[model_name] = ModelRuntimeState(
                resolved_backend=backend,
                configured_enabled=model_settings.enabled,
            )
            if not model_settings.enabled:
                continue
            grouped_models.setdefault(backend, {})[model_name] = model_settings

        for backend, models in grouped_models.items():
            scoped_settings = replace(
                settings,
                engine=replace(
                    settings.engine,
                    backend=backend,
                    models=models,
                ),
            )
            try:
                backend_engine = self._build_backend_engine(backend, scoped_settings)
            except Exception as exc:
                message = _exception_message(exc)
                for model_name in models:
                    state = self._model_states[model_name]
                    state.lifecycle = "failed"
                    state.last_error = message
                LOGGER.exception(
                    "Failed to initialize backend '%s'; skipping %d model(s).",
                    backend,
                    len(models),
                )
                continue
            loaded_models = getattr(backend_engine, "_models", {})
            load_errors = getattr(backend_engine, "_load_errors", {})
            for model_name in models:
                state = self._model_states[model_name]
                if model_name in loaded_models:
                    state.lifecycle = "loaded"
                    state.last_error = None
                else:
                    state.lifecycle = "failed"
                    state.last_error = str(load_errors.get(model_name, "model failed to load"))
            if not loaded_models:
                LOGGER.warning(
                    "Backend '%s' initialized without loaded models; skipping backend.",
                    backend,
                )
                continue
            for model_name, runtime in loaded_models.items():
                self._models[model_name] = runtime
                self._model_engines[model_name] = backend_engine

    def complete(self, request: ResponseRequest) -> EngineResult:
        with self._state_lock:
            if request.model not in self._configured_models:
                raise UnknownModelError(request.model)
            state = self._model_states[request.model]
            if state.lifecycle != "loaded":
                raise ModelStateError(request.model, self._lifecycle_error_code(state.lifecycle))
            engine = self._model_engines.get(request.model)
            if engine is None:
                raise ModelStateError(request.model, "model_not_loaded")
            state.inflight_requests += 1
        try:
            return engine.complete(request)
        finally:
            with self._state_lock:
                state = self._model_states.get(request.model)
                if state is not None and state.inflight_requests > 0:
                    state.inflight_requests -= 1
                    self._state_changed.notify_all()

    def admin_models_payload(self, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        with self._state_lock:
            models: list[dict[str, object]] = []
            for model_name, model_settings in self._configured_models.items():
                models.append(self._admin_model_entry_locked(model_name, model_settings))
            return {"models": models}

    def load_model(self, model_name: str, settings: AppSettings | None = None) -> dict[str, object]:
        if settings is None:
            raise RuntimeError("settings are required to load a model")

        with self._state_lock:
            model_settings = self._configured_models.get(model_name)
            if model_settings is None:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle == "unloading":
                raise ModelStateError(model_name, "model_unloading")
            if state.lifecycle in {"loaded", "loading"}:
                return self._admin_model_entry_locked(model_name, model_settings)
            state.lifecycle = "loading"
            state.last_error = None
            resolved_backend = state.resolved_backend

        scoped_settings = replace(
            settings,
            engine=replace(
                settings.engine,
                backend=resolved_backend,
                models={model_name: model_settings},
            ),
        )

        try:
            backend_engine = self._build_backend_engine(resolved_backend, scoped_settings)
        except Exception as exc:
            message = _exception_message(exc)
            with self._state_lock:
                state = self._model_states[model_name]
                state.lifecycle = "failed"
                state.last_error = message
            LOGGER.exception(
                "Failed to load model '%s' from %s.",
                model_name,
                model_settings.model_path,
            )
            raise RuntimeError(message) from exc

        loaded_models = getattr(backend_engine, "_models", {})
        load_errors = getattr(backend_engine, "_load_errors", {})
        runtime = loaded_models.get(model_name)
        if runtime is None:
            message = str(load_errors.get(model_name, "model failed to load"))
            with self._state_lock:
                state = self._model_states[model_name]
                state.lifecycle = "failed"
                state.last_error = message
            raise RuntimeError(message)

        with self._state_lock:
            existing_engine = self._find_backend_engine_locked(resolved_backend)
            target_engine = existing_engine or backend_engine
            if existing_engine is not None:
                existing_engine._models[model_name] = runtime
                if hasattr(existing_engine, "_load_errors"):
                    existing_engine._load_errors.pop(model_name, None)
            self._models[model_name] = runtime
            self._model_engines[model_name] = target_engine
            state = self._model_states[model_name]
            state.lifecycle = "loaded"
            state.last_error = None
            return self._admin_model_entry_locked(model_name, model_settings)

    def unload_model(self, model_name: str, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        with self._state_lock:
            model_settings = self._configured_models.get(model_name)
            if model_settings is None:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle == "loading":
                raise ModelStateError(model_name, "model_loading")
            if state.lifecycle in {"unloaded", "failed", "unloading"}:
                return self._admin_model_entry_locked(model_name, model_settings)

            state.lifecycle = "unloading"
            while state.inflight_requests > 0:
                self._state_changed.wait()

            runtime = self._models.pop(model_name, None)
            backend_engine = self._model_engines.pop(model_name, None)
            if backend_engine is not None:
                backend_engine._models.pop(model_name, None)
                if hasattr(backend_engine, "_load_errors"):
                    backend_engine._load_errors.pop(model_name, None)

        self._cleanup_runtime(runtime)

        with self._state_lock:
            state = self._model_states[model_name]
            state.lifecycle = "unloaded"
            state.last_error = None
            self._state_changed.notify_all()
            return self._admin_model_entry_locked(model_name, model_settings)

    def _admin_model_entry_locked(self, model_name: str, model_settings: ModelSettings) -> dict[str, object]:
        state = self._model_states[model_name]
        return {
            "name": model_name,
            "resolved_backend": state.resolved_backend,
            "configured_enabled": state.configured_enabled,
            "runtime_state": state.lifecycle,
            "is_loaded": state.lifecycle == "loaded",
            "inflight_requests": state.inflight_requests,
            "last_error": state.last_error,
            "definition": asdict(model_settings),
        }

    def _find_backend_engine_locked(self, backend: str):
        for loaded_model_name, engine in self._model_engines.items():
            state = self._model_states.get(loaded_model_name)
            if state is not None and state.resolved_backend == backend:
                return engine
        return None

    def _cleanup_runtime(self, runtime: object | None) -> None:
        if runtime is None:
            gc.collect()
            return

        generator = getattr(runtime, "generator", None)
        clear_queue = getattr(generator, "clear_queue", None)
        if callable(clear_queue):
            try:
                clear_queue()
            except Exception:
                LOGGER.warning("Failed to clear backend queue during runtime cleanup.", exc_info=True)

        llm = getattr(runtime, "llm", None)
        close = getattr(llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.warning("Failed to close GGUF runtime during cleanup.", exc_info=True)

        gc.collect()

    def _lifecycle_error_code(self, lifecycle: str) -> str:
        if lifecycle == "loading":
            return "model_loading"
        if lifecycle == "unloading":
            return "model_unloading"
        if lifecycle == "failed":
            return "model_failed"
        return "model_not_loaded"

    def _resolve_model_backend(self, default_backend: str, model_settings: ModelSettings) -> str:
        backend = model_settings.backend or default_backend
        backend = backend.strip().lower()
        if backend == "":
            raise ValueError("backend cannot be empty")
        return backend

    def _build_backend_engine(self, backend: str, settings: AppSettings):
        if backend == "ct2":
            return Ct2Engine(settings)
        if backend == "exllamav3":
            return ExLlamaV3Engine(settings)
        if backend == "gguf":
            return LlamaCppEngine(settings)
        raise ValueError(f"unsupported engine backend: {backend!r}")


def build_engine(settings: AppSettings):
    if settings.engine.backend == "stub":
        return StubEngine(settings)
    return ModelRouterEngine(settings)
