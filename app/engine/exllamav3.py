from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
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
from .common import _resolve_request_enable_thinking
from .common import ResolvedDecoding


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

        user_text = _require_text_input(request)
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
            user_text=user_text,
            enable_thinking=_resolve_request_enable_thinking(
                request,
                runtime.config.enable_thinking,
            ),
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
