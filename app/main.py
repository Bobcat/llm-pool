from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.config import load_settings
from app.engine import build_engine
from app.schemas import OutputText
from app.schemas import ResponseEnvelope
from app.schemas import ResponseRequest


def _chunk_text(text: str, *, size: int = 24) -> list[str]:
    if text == "":
        return [""]
    return [text[index : index + size] for index in range(0, len(text), size)]


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _stream_response(response_id: str, request: ResponseRequest, output_text: str) -> Iterator[str]:
    yield _sse_event(
        "response.created",
        {"id": response_id, "model": request.model, "object": "response"},
    )
    for chunk in _chunk_text(output_text):
        yield _sse_event(
            "response.output_text.delta",
            {"id": response_id, "delta": chunk},
        )
    yield _sse_event(
        "response.completed",
        {"id": response_id, "output_text": output_text},
    )

def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    engine = build_engine(settings)
    app = FastAPI(title="LLM Pool API")

    @app.post("/v1/responses")
    def create_response(request: ResponseRequest):
        response_id = f"resp_{uuid.uuid4().hex}"
        result = engine.complete(request)
        if request.stream:
            return StreamingResponse(
                _stream_response(response_id, request, result.text),
                media_type="text/event-stream",
            )

        return ResponseEnvelope(
            id=response_id,
            model=request.model,
            output=[OutputText(text=result.text)],
            output_text=result.text,
        )

    return app


app = create_app()
