from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.config import load_settings
from app.engine import build_engine
from app.engine import ModelStateError
from app.engine import UnknownModelError
from app.schemas import AdminModelEntry
from app.schemas import AdminGpuMemoryEnvelope
from app.schemas import AdminLoadRequest
from app.schemas import AdminModelsEnvelope
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

    @app.on_event("shutdown")
    def shutdown_engine() -> None:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    @app.get("/v1/models")
    def list_models() -> dict[str, object]:
        list_models_payload = getattr(engine, "list_models_payload", None)
        if callable(list_models_payload):
            return list_models_payload()
        loaded_models = sorted(getattr(engine, "_models", {}).keys())
        return {"models": loaded_models}

    @app.get(
        "/v1/admin/models",
        response_model=AdminModelsEnvelope,
        tags=["admin"],
        summary="List configured models and runtime state",
        description=(
            "Returns all models from the merged settings plus their current in-process "
            "runtime state. This is the primary read-only admin endpoint for a future UI."
        ),
    )
    def list_admin_models() -> dict[str, object]:
        return engine.admin_models_payload(settings)

    @app.get(
        "/v1/admin/gpu-memory",
        response_model=AdminGpuMemoryEnvelope,
        tags=["admin"],
        summary="Get GPU memory usage and model VRAM estimates",
        description=(
            "Returns current GPU memory usage from nvidia-smi and approximate per-model VRAM usage "
            "for both loaded and unloaded models."
        ),
    )
    def get_gpu_memory() -> dict[str, object]:
        return engine.admin_gpu_memory_payload(settings)

    @app.post(
        "/v1/admin/models/{model_name}/load",
        response_model=AdminModelEntry,
        tags=["admin"],
        summary="Load one configured model",
        description=(
            "Loads one model that already exists in the merged settings. "
            "This operation is live-only and does not modify settings files. "
            "An optional request body may supply temporary backend-specific load overrides."
        ),
    )
    def load_model(model_name: str, load_request: AdminLoadRequest | None = None) -> dict[str, object]:
        try:
            return engine.load_model(model_name, settings, load_request)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_model", "model": model_name},
            ) from exc
        except ModelStateError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.code, "model": model_name},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_load_request",
                    "model": model_name,
                    "message": str(exc),
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "model_load_failed",
                    "model": model_name,
                    "message": str(exc),
                },
            ) from exc

    @app.post(
        "/v1/admin/models/{model_name}/unload",
        response_model=AdminModelEntry,
        tags=["admin"],
        summary="Unload one loaded model",
        description=(
            "Gracefully unloads one model at runtime. New requests are rejected once unloading "
            "starts, and in-flight requests are allowed to finish before resources are released."
        ),
    )
    def unload_model(model_name: str) -> dict[str, object]:
        try:
            return engine.unload_model(model_name, settings)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_model", "model": model_name},
            ) from exc
        except ModelStateError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.code, "model": model_name},
            ) from exc

    @app.post("/v1/responses")
    def create_response(request: ResponseRequest):
        started_at = time.perf_counter()
        response_id = f"resp_{uuid.uuid4().hex}"
        try:
            result = engine.complete(request)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_model", "model": request.model},
            ) from exc
        except ModelStateError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.code, "model": request.model},
            ) from exc
        total_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        metrics_payload = (
            result.metrics.model_dump()
            if hasattr(result.metrics, "model_dump")
            else result.metrics.dict()
        )
        metrics_payload["pool_total_wall_ms"] = total_ms
        metrics = ResponseMetrics(**metrics_payload)
        _log_inference(response_id, request, metrics)
        if request.stream:
            return StreamingResponse(
                _stream_response(response_id, request, result.text, metrics),
                media_type="text/event-stream",
            )

        return ResponseEnvelope(
            id=response_id,
            model=request.model,
            output=[OutputText(text=result.text)],
            output_text=result.text,
            metrics=metrics,
        )

    return app


app = create_app()
