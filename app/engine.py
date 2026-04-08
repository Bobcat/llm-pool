from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import EngineResult
from app.schemas import ResponseRequest


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


class Ct2Engine:
    def __init__(self, settings: AppSettings) -> None:
        self.default_model = settings.engine.default_model
        self._models: dict[str, Ct2ModelRuntime] = {}
        for model_name, model_settings in settings.engine.models.items():
            self._models[model_name] = self._build_runtime(model_settings)
        if self.default_model not in self._models:
            raise ValueError(f"default model {self.default_model!r} is not configured")

    def complete(self, request: ResponseRequest) -> EngineResult:
        runtime = self._models.get(request.model)
        if runtime is None:
            raise ValueError(f"unknown model: {request.model!r}")

        system_prompt = request.instructions or "You are a helpful assistant. Return only the response."
        user_text = f"{request.input}<|im_end|>\n<|im_start|>assistant\n"
        request_tokens = self._tokenize(runtime.tokenizer, user_text, add_special_tokens=False)
        result = runtime.generator.generate_batch(  # type: ignore[call-arg]
            [request_tokens],
            static_prompt=self._get_static_prompt_tokens(runtime, system_prompt),
            cache_static_prompt=True,
            include_prompt_in_result=False,
            beam_size=request.decoding.beam_size,
            max_length=request.decoding.max_tokens,
            sampling_topk=request.decoding.top_k,
            sampling_topp=request.decoding.top_p,
            sampling_temperature=request.decoding.temperature,
            repetition_penalty=request.decoding.repetition_penalty,
            end_token=request.decoding.stop[0] if request.decoding.stop else "<|im_end|>",
        )
        if not result or not result[0].sequences:
            return EngineResult(text="")
        return EngineResult(text=self._decode(runtime.tokenizer, result[0].sequences[0]).strip())

    def _build_runtime(self, settings: ModelSettings) -> Ct2ModelRuntime:
        try:
            import ctranslate2
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("ctranslate2 and transformers are required for the CT2 engine") from exc

        model_path = Path(settings.model_path)
        generator = ctranslate2.Generator(
            str(model_path),
            device=settings.device,
            compute_type=settings.compute_type,
        )
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

    def _decode(self, tokenizer: object, tokens: list[str]) -> str:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_engine(settings: AppSettings):
    if settings.engine.backend == "stub":
        return StubEngine()
    if settings.engine.backend == "ct2":
        return Ct2Engine(settings)
    raise ValueError(f"unsupported engine backend: {settings.engine.backend!r}")
