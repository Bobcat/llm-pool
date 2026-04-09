from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class DecodingParams(BaseModel):
    beam_size: int | None = Field(default=None, ge=1, le=16)
    top_k: int | None = Field(default=None, ge=1, le=200)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=4.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stop: list[str] | None = None


class ResponseRequest(BaseModel):
    model: str
    input: str
    instructions: str | None = None
    stream: bool = False
    decoding: DecodingParams = Field(default_factory=DecodingParams)


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseMetrics(BaseModel):
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


@dataclass(frozen=True)
class EngineResult:
    text: str
    metrics: ResponseMetrics = field(default_factory=ResponseMetrics)
