from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
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
from .common import ResolvedDecoding


@dataclass
class Ct2ModelRuntime:
    config: ModelSettings
    generator: object
    tokenizer: object
    prompt_token_cache: dict[str, list[str]] = field(default_factory=dict)


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
