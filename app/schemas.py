from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class ModalityUnsupportedError(ValueError):
    """Raised when image content is passed to a text-only path."""


class DecodingParams(BaseModel):
    beam_size: int | None = Field(default=None, ge=1, le=16)
    top_k: int | None = Field(default=None, ge=1, le=200)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=4.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stop: list[str] | None = None


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlSpec(BaseModel):
    url: str
    detail: Literal["low", "high", "auto"] = "auto"


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSpec


ContentItem = Annotated[
    Union[TextContent, ImageContent],
    Field(discriminator="type"),
]


class Message(BaseModel):
    """A single turn in a multi-turn conversation.

    The system prompt is carried by ``ResponseRequest.instructions``, so only
    dialogue roles appear here. ``content`` mirrors ``input``: plain text or a
    list of text/image content items.
    """

    role: Literal["user", "assistant"]
    content: str | list[ContentItem]


class ResponseRequest(BaseModel):
    model: str
    input: str | list[ContentItem] | None = None
    messages: list[Message] | None = None
    instructions: str | None = None
    source_lang_code: str | None = None
    target_lang_code: str | None = None
    allow_remote: bool = False
    stream: bool = False
    decoding: DecodingParams = Field(default_factory=DecodingParams)

    @model_validator(mode="after")
    def _require_input_or_messages(self) -> "ResponseRequest":
        if self.messages is not None:
            if len(self.messages) == 0:
                raise ValueError("messages must not be empty")
            if self.messages[-1].role != "user":
                raise ValueError("the last message must have role 'user'")
        elif self.input is None:
            raise ValueError("provide either 'input' or 'messages'")
        return self

    @property
    def has_image_content(self) -> bool:
        if self.input is None or isinstance(self.input, str):
            return False
        return any(isinstance(item, ImageContent) for item in self.input)

    def text_input_or_raise(self) -> str:
        """Return the request input flattened to plain text.

        For string input, returns it unchanged (backwards-compat).
        For list input, joins all TextContent items. Raises
        ``ModalityUnsupportedError`` if any ImageContent is present.
        """
        if self.input is None:
            raise ModalityUnsupportedError(
                "this backend requires 'input'; multi-turn 'messages' is not supported"
            )
        if isinstance(self.input, str):
            return self.input
        text_parts: list[str] = []
        for item in self.input:
            if isinstance(item, ImageContent):
                raise ModalityUnsupportedError(
                    "image content is not supported by this model"
                )
            text_parts.append(item.text)
        return "".join(text_parts)


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseMetrics(BaseModel):
    engine_queue_wait_ms: float | None = None
    backend_inference_wall_ms: float | None = None
    engine_total_wall_ms: float | None = None
    engine_outside_backend_wall_ms: float | None = None
    pool_total_wall_ms: float | None = None
    engine_tokenize_ms: float | None = None
    gpu_time_to_first_token_ms: float | None = None
    gpu_generate_total_ms: float | None = None
    gpu_decode_after_first_token_ms: float | None = None
    engine_prompt_tokens: int | None = None
    engine_output_tokens: int | None = None
    engine_tokens_per_second: float | None = None


class ResponseEnvelope(BaseModel):
    id: str
    object: Literal["response"] = "response"
    model: str
    output: list[OutputText]
    output_text: str
    metrics: ResponseMetrics = Field(default_factory=ResponseMetrics)


class AdminLoadRequest(BaseModel):
    replicas: int | None = Field(default=None, ge=1)
    gguf_n_ctx: int | None = Field(default=None, ge=1)
    gguf_flash_attn: str | None = None
    gguf_type_k: str | None = None
    gguf_type_v: str | None = None
    exllama_cache_size: int | None = Field(default=None, ge=256)
    exllama_cache_k_bits: int | None = Field(default=None, ge=2, le=8)
    exllama_cache_v_bits: int | None = Field(default=None, ge=2, le=8)
    exllama_cache_quant: str | None = None
    exllama_max_rq_tokens: int | None = Field(default=None, ge=1)
    vllm_max_model_len: int | None = Field(default=None, ge=256)
    vllm_kv_cache_dtype: str | None = None
    vllm_kv_cache_memory_bytes: int | None = Field(default=None, ge=1)
    vllm_max_pixels: int | None = Field(default=None, ge=1)


class ModelCapabilities(BaseModel):
    modalities: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    multi_turn: bool = False


class AdminModelEntry(BaseModel):
    name: str
    resolved_backend: str
    configured_enabled: bool
    runtime_state: Literal["unloaded", "loading", "loaded", "unloading", "failed"]
    is_loaded: bool
    replicas: int = Field(default=1, ge=1)
    replica_max: int = Field(default=1, ge=1)
    loaded_replicas: int = Field(default=0, ge=0)
    inflight_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    runtime_inflight: int = Field(default=0, ge=0)
    configured_target_inflight: int = Field(default=1, ge=1)
    effective_target_inflight: int = Field(default=1, ge=1)
    last_error: str | None = None
    vram_estimate_mib: int | None = Field(default=None, ge=0)
    vram_estimate_replica_count: int | None = Field(default=None, ge=1)
    vram_estimate_source: Literal["observed_load_delta", "model_artifact_size", "unavailable"] = "unavailable"
    load_constraints: dict[str, Any] = Field(default_factory=dict)
    load_recommendations: dict[str, Any] = Field(default_factory=dict)
    load_override: dict[str, Any] = Field(default_factory=dict)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    definition: dict[str, Any] = Field(default_factory=dict)


class AdminModelsEnvelope(BaseModel):
    models: list[AdminModelEntry] = Field(default_factory=list)


class AdminGpuMemoryDevice(BaseModel):
    index: int
    name: str
    used_mib: int = Field(ge=0)
    total_mib: int = Field(ge=0)
    used_over_total: str


class AdminGpuMemoryModelEstimate(BaseModel):
    name: str
    runtime_state: Literal["unloaded", "loading", "loaded", "unloading", "failed"]
    is_loaded: bool
    configured_target_inflight: int = Field(default=1, ge=1)
    effective_target_inflight: int = Field(default=1, ge=1)
    vram_estimate_mib: int | None = Field(default=None, ge=0)
    vram_estimate_replica_count: int | None = Field(default=None, ge=1)
    vram_estimate_source: Literal["observed_load_delta", "model_artifact_size", "unavailable"] = "unavailable"


class AdminGpuMemoryEnvelope(BaseModel):
    gpus: list[AdminGpuMemoryDevice] = Field(default_factory=list)
    models: list[AdminGpuMemoryModelEstimate] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class EngineResult:
    text: str
    metrics: ResponseMetrics = field(default_factory=ResponseMetrics)
