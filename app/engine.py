from __future__ import annotations

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


class StubEngine:
    """Temporary engine used to validate the API contract before model integration."""

    def complete(self, request: ResponseRequest) -> EngineResult:
        text = request.input
        if request.instructions:
            text = f"[instructions={request.instructions}] {text}"
        return EngineResult(text=text)


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
        self.default_model = settings.engine.default_model
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, Ct2ModelRuntime] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_settings)
            except Exception:
                LOGGER.exception(
                    "Failed to load model '%s' from %s; skipping model.",
                    model_name,
                    model_settings.model_path,
                )
        if self.default_model not in self._models:
            loaded_models = list(self._models.keys())
            if not loaded_models:
                raise ValueError("no enabled models could be loaded")
            fallback_default = loaded_models[0]
            LOGGER.warning(
                "Default model '%s' could not be loaded; falling back to '%s'.",
                self.default_model,
                fallback_default,
            )
            self.default_model = fallback_default

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
        self.default_model = settings.engine.default_model
        self.decoding_defaults = settings.engine.decoding
        self._models: dict[str, ExLlamaV3ModelRuntime] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            try:
                self._models[model_name] = self._build_runtime(model_settings)
            except Exception:
                LOGGER.exception(
                    "Failed to load model '%s' from %s; skipping model.",
                    model_name,
                    model_settings.model_path,
                )
        if self.default_model not in self._models:
            loaded_models = list(self._models.keys())
            if not loaded_models:
                raise ValueError("no enabled models could be loaded")
            fallback_default = loaded_models[0]
            LOGGER.warning(
                "Default model '%s' could not be loaded; falling back to '%s'.",
                self.default_model,
                fallback_default,
            )
            self.default_model = fallback_default

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


class ModelRouterEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.default_model = settings.engine.default_model
        self._models: dict[str, object] = {}
        self._model_engines: dict[str, object] = {}
        grouped_models: dict[str, dict[str, ModelSettings]] = {}
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            backend = self._resolve_model_backend(settings.engine.backend, model_settings)
            grouped_models.setdefault(backend, {})[model_name] = model_settings

        if not grouped_models:
            raise ValueError("no enabled models could be loaded")

        for backend, models in grouped_models.items():
            scoped_default_model = (
                settings.engine.default_model
                if settings.engine.default_model in models
                else next(iter(models.keys()))
            )
            scoped_settings = replace(
                settings,
                engine=replace(
                    settings.engine,
                    backend=backend,
                    default_model=scoped_default_model,
                    models=models,
                ),
            )
            try:
                backend_engine = self._build_backend_engine(backend, scoped_settings)
            except Exception:
                LOGGER.exception(
                    "Failed to initialize backend '%s'; skipping %d model(s).",
                    backend,
                    len(models),
                )
                continue
            loaded_models = getattr(backend_engine, "_models", {})
            if not loaded_models:
                LOGGER.warning(
                    "Backend '%s' initialized without loaded models; skipping backend.",
                    backend,
                )
                continue
            for model_name, runtime in loaded_models.items():
                self._models[model_name] = runtime
                self._model_engines[model_name] = backend_engine

        if not self._model_engines:
            raise ValueError("no enabled models could be loaded")

        if self.default_model not in self._model_engines:
            fallback_default = sorted(self._model_engines.keys())[0]
            LOGGER.warning(
                "Default model '%s' could not be loaded; falling back to '%s'.",
                self.default_model,
                fallback_default,
            )
            self.default_model = fallback_default

    def complete(self, request: ResponseRequest) -> EngineResult:
        engine = self._model_engines.get(request.model)
        if engine is None:
            raise ValueError(f"unknown model: {request.model!r}")
        return engine.complete(request)

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
        raise ValueError(f"unsupported engine backend: {backend!r}")


def build_engine(settings: AppSettings):
    if settings.engine.backend == "stub":
        return StubEngine()

    has_model_backend_overrides = any(
        model_settings.backend is not None
        for model_settings in settings.engine.models.values()
    )
    if has_model_backend_overrides:
        return ModelRouterEngine(settings)
    if settings.engine.backend == "ct2":
        return Ct2Engine(settings)
    if settings.engine.backend == "exllamav3":
        return ExLlamaV3Engine(settings)
    raise ValueError(f"unsupported engine backend: {settings.engine.backend!r}")
