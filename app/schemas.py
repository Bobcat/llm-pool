from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
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


class AudioUrlSpec(BaseModel):
    url: str


class AudioContent(BaseModel):
    type: Literal["audio_url"] = "audio_url"
    audio_url: AudioUrlSpec


class FileSpec(BaseModel):
    filename: str = Field(min_length=1)
    file_data: str = Field(min_length=1)


class FileContent(BaseModel):
    type: Literal["file"] = "file"
    file: FileSpec


ContentItem = Annotated[
    Union[TextContent, ImageContent, AudioContent, FileContent],
    Field(discriminator="type"),
]
ThinkingMode = Literal["default", "enabled", "disabled"]
ResponseFormatName = Literal["text", "json_schema"]


class JsonSchemaDefinition(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    schema_definition: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class JsonSchemaResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


class Message(BaseModel):
    """A single turn in a multi-turn conversation.

    The system prompt is carried by ``ResponseRequest.instructions``, so only
    dialogue roles appear here. ``content`` mirrors ``input``: plain text or a
    list of text/image/audio content items.
    """

    role: Literal["user", "assistant"]
    content: str | list[ContentItem]


class ResponseRequest(BaseModel):
    model: str
    input: str | list[ContentItem] | None = None
    messages: list[Message] | None = None
    instructions: str | None = None
    fairness_key: str | None = Field(default=None, max_length=128)
    source_lang_code: str | None = Field(
        default=None,
        description=(
            "Source language code for translation models. For llama_cpp TranslateGemma, "
            "omit this field or use 'auto'/'mixed' to request mixed-source detection."
        ),
    )
    target_lang_code: str | None = Field(
        default=None,
        description="Target language code for translation models.",
    )
    allow_remote: bool = False
    prompt_cache_key: str | None = Field(default=None, min_length=1)
    stream: bool = False
    thinking: ThinkingMode = "default"
    response_format: JsonSchemaResponseFormat | None = None
    decoding: DecodingParams = Field(default_factory=DecodingParams)

    @field_validator("fairness_key", mode="before")
    @classmethod
    def _normalize_fairness_key(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("fairness_key must be a string")
        normalized = value.strip()
        if normalized == "":
            raise ValueError("fairness_key must not be blank")
        return normalized

    @model_validator(mode="after")
    def _require_input_or_messages(self) -> "ResponseRequest":
        if self.response_format is not None and self.stream:
            raise ValueError("response_format is not supported with stream=true")
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
        return any(
            isinstance(item, ImageContent)
            for item in self._content_items()
        )

    @property
    def has_audio_content(self) -> bool:
        return any(
            isinstance(item, AudioContent)
            for item in self._content_items()
        )

    @property
    def has_file_content(self) -> bool:
        return any(
            isinstance(item, FileContent)
            for item in self._content_items()
        )

    def _content_items(self) -> list[ContentItem]:
        items: list[ContentItem] = []
        if isinstance(self.input, list):
            items.extend(self.input)
        for message in self.messages or []:
            if isinstance(message.content, list):
                items.extend(message.content)
        return items

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
            if isinstance(item, (ImageContent, AudioContent, FileContent)):
                raise ModalityUnsupportedError(
                    f"{item.type} content is not supported by this model"
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
    engine_cached_prompt_tokens: int | None = None
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
    vllm_speculative_method: str | None = None
    vllm_speculative_model: str | None = None
    vllm_speculative_moe_backend: str | None = None
    vllm_speculative_attention_backend: str | None = None
    vllm_num_speculative_tokens: int | None = Field(default=None, ge=1)
    llama_server_n_ctx: int | None = Field(default=None, ge=1)
    llama_server_image_max_tokens: int | None = Field(default=None, ge=1)
    llama_server_spec_type: str | None = None
    llama_server_spec_draft_n_max: int | None = Field(default=None, ge=1, le=6)
    llama_server_spec_draft_p_min: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelCapabilities(BaseModel):
    modalities: list[Literal["text", "image", "audio"]] = Field(
        default_factory=lambda: ["text"]
    )
    file_inputs: bool = False
    multi_turn: bool = False
    thinking_modes: list[ThinkingMode] = Field(default_factory=lambda: ["default"])
    response_formats: list[ResponseFormatName] = Field(default_factory=lambda: ["text"])


class AdminFairnessKeyEntry(BaseModel):
    fairness_key: str | None = None
    pending: int = Field(default=0, ge=0)
    active: int = Field(default=0, ge=0)
    weight: float = Field(default=1.0, gt=0.0)
    score: float = Field(default=0.0, ge=0.0)
    rejected_per_key_limit: int = Field(default=0, ge=0)
    rejected_executor_limit: int = Field(default=0, ge=0)


class AdminFairnessSnapshot(BaseModel):
    rejected_per_key_limit: int = Field(default=0, ge=0)
    rejected_executor_limit: int = Field(default=0, ge=0)
    keys: list[AdminFairnessKeyEntry] = Field(default_factory=list)


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
    fairness: AdminFairnessSnapshot = Field(default_factory=AdminFairnessSnapshot)
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


class AdminSettingsReloadEnvelope(BaseModel):
    added_models: list[str] = Field(default_factory=list)
    removed_models: list[str] = Field(default_factory=list)
    updated_models: list[str] = Field(default_factory=list)
    unchanged_models: list[str] = Field(default_factory=list)
    service_restart_required: bool = False


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
