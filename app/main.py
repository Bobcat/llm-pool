from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.config import load_settings
from app.engine import build_engine
from app.schemas import OutputText
from app.schemas import ResponseEnvelope
from app.schemas import ResponseMetrics
from app.schemas import ResponseRequest


LOGGER = logging.getLogger("llm_pool.metrics")


def _chunk_text(text: str, *, size: int = 24) -> list[str]:
    if text == "":
        return [""]
    return [text[index : index + size] for index in range(0, len(text), size)]


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _metrics_payload(metrics: ResponseMetrics) -> dict[str, object]:
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    return metrics.dict()


def _stream_response(
    response_id: str,
    request: ResponseRequest,
    output_text: str,
    metrics: ResponseMetrics,
) -> Iterator[str]:
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
        "response.metrics",
        {"id": response_id, "metrics": _metrics_payload(metrics)},
    )
    yield _sse_event(
        "response.completed",
        {"id": response_id, "output_text": output_text},
    )


def _log_inference(response_id: str, request: ResponseRequest, metrics: ResponseMetrics) -> None:
    payload = {
        "event": "llm_pool.inference",
        "request_id": response_id,
        "model": request.model,
        "stream": request.stream,
        "metrics": _metrics_payload(metrics),
    }
    LOGGER.info("%s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    engine = build_engine(settings)
    app = FastAPI(title="LLM Pool API")

    @app.get("/v1/models")
    def list_models() -> dict[str, object]:
        loaded_models = sorted(getattr(engine, "_models", {}).keys())
        if not loaded_models:
            loaded_models = [
                model_name
                for model_name, model_settings in settings.engine.models.items()
                if model_settings.enabled
            ]
        default_model = getattr(engine, "default_model", settings.engine.default_model)
        return {
            "default_model": default_model,
            "models": loaded_models,
        }

    @app.post("/v1/responses")
    def create_response(request: ResponseRequest):
        response_id = f"resp_{uuid.uuid4().hex}"
        result = engine.complete(request)
        _log_inference(response_id, request, result.metrics)
        if request.stream:
            return StreamingResponse(
                _stream_response(response_id, request, result.text, result.metrics),
                media_type="text/event-stream",
            )

        return ResponseEnvelope(
            id=response_id,
            model=request.model,
            output=[OutputText(text=result.text)],
            output_text=result.text,
            metrics=result.metrics,
        )

    return app


app = create_app()
