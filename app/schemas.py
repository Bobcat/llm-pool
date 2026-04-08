from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class DecodingParams(BaseModel):
    beam_size: int = Field(default=1, ge=1, le=16)
    top_k: int = Field(default=1, ge=1, le=200)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=4.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stop: list[str] = Field(default_factory=lambda: ["<|im_end|>"])


class ResponseRequest(BaseModel):
    model: str
    input: str
    instructions: str | None = None
    stream: bool = False
    decoding: DecodingParams = Field(default_factory=DecodingParams)


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseEnvelope(BaseModel):
    id: str
    object: Literal["response"] = "response"
    model: str
    output: list[OutputText]
    output_text: str


@dataclass(frozen=True)
class EngineResult:
    text: str
