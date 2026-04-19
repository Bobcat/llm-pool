from __future__ import annotations

from app.config import AppSettings
from app.schemas import AdminLoadRequest
from app.schemas import EngineResult
from app.schemas import ResponseRequest

from .common import _estimate_model_artifact_size_mib
from .common import _load_constraints_for_backend
from .common import _load_recommendations_for_backend
from .common import _model_definition_payload


class StubEngine:
    """Temporary engine used to validate the API contract before model integration."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._configured_models = dict(settings.engine.models) if settings is not None else {}
        self._models = {
            model_name: object()
            for model_name, model_settings in self._configured_models.items()
            if model_settings.enabled
        }

    def complete(self, request: ResponseRequest) -> EngineResult:
        if request.model not in self._configured_models:
            from .common import UnknownModelError

            raise UnknownModelError(request.model)
        if request.model not in self._models:
            from .common import ModelStateError

            raise ModelStateError(request.model, "model_not_loaded")
        text = request.input
        if request.instructions:
            text = f"[instructions={request.instructions}] {text}"
        return EngineResult(text=text)

    def admin_models_payload(self, settings: AppSettings) -> dict[str, object]:
        models: list[dict[str, object]] = []
        for model_name, model_settings in self._configured_models.items():
            is_loaded = model_name in self._models
            vram_estimate_mib = _estimate_model_artifact_size_mib(model_settings.model_path)
            models.append(
                {
                    "name": model_name,
                    "resolved_backend": settings.engine.backend,
                    "configured_enabled": model_settings.enabled,
                    "runtime_state": "loaded" if is_loaded else "unloaded",
                    "is_loaded": is_loaded,
                    "inflight_requests": 0,
                    "last_error": None,
                    "vram_estimate_mib": vram_estimate_mib,
                    "vram_estimate_source": (
                        "model_artifact_size" if vram_estimate_mib is not None else "unavailable"
                    ),
                    "load_constraints": _load_constraints_for_backend(settings.engine.backend),
                    "load_recommendations": _load_recommendations_for_backend(settings.engine.backend),
                    "load_override": {},
                    "definition": _model_definition_payload(
                        model_settings,
                        resolved_backend=settings.engine.backend,
                    ),
                }
            )
        return {"models": models}

    def admin_gpu_memory_payload(self, settings: AppSettings) -> dict[str, object]:
        from .common import _query_gpu_memory

        gpus, error = _query_gpu_memory()
        models: list[dict[str, object]] = []
        for model_name, model_settings in self._configured_models.items():
            is_loaded = model_name in self._models
            vram_estimate_mib = _estimate_model_artifact_size_mib(model_settings.model_path)
            models.append(
                {
                    "name": model_name,
                    "runtime_state": "loaded" if is_loaded else "unloaded",
                    "is_loaded": is_loaded,
                    "vram_estimate_mib": vram_estimate_mib,
                    "vram_estimate_source": (
                        "model_artifact_size" if vram_estimate_mib is not None else "unavailable"
                    ),
                }
            )
        return {"gpus": gpus, "models": models, "error": error}

    def load_model(
        self,
        model_name: str,
        settings: AppSettings,
        load_request: AdminLoadRequest | None = None,
    ) -> dict[str, object]:
        from .common import UnknownModelError

        model_settings = self._configured_models.get(model_name)
        if model_settings is None:
            raise UnknownModelError(model_name)
        if self._has_load_overrides(load_request):
            raise ValueError("load overrides are not supported for the stub backend")
        self._models.setdefault(model_name, object())
        vram_estimate_mib = _estimate_model_artifact_size_mib(model_settings.model_path)
        return {
            "name": model_name,
            "resolved_backend": settings.engine.backend,
            "configured_enabled": model_settings.enabled,
            "runtime_state": "loaded",
            "is_loaded": True,
            "inflight_requests": 0,
            "last_error": None,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": "model_artifact_size" if vram_estimate_mib is not None else "unavailable",
            "load_constraints": _load_constraints_for_backend(settings.engine.backend),
            "load_recommendations": _load_recommendations_for_backend(settings.engine.backend),
            "load_override": {},
            "definition": _model_definition_payload(
                model_settings,
                resolved_backend=settings.engine.backend,
            ),
        }

    def unload_model(self, model_name: str, settings: AppSettings) -> dict[str, object]:
        from .common import UnknownModelError

        model_settings = self._configured_models.get(model_name)
        if model_settings is None:
            raise UnknownModelError(model_name)
        self._models.pop(model_name, None)
        vram_estimate_mib = _estimate_model_artifact_size_mib(model_settings.model_path)
        return {
            "name": model_name,
            "resolved_backend": settings.engine.backend,
            "configured_enabled": model_settings.enabled,
            "runtime_state": "unloaded",
            "is_loaded": False,
            "inflight_requests": 0,
            "last_error": None,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": "model_artifact_size" if vram_estimate_mib is not None else "unavailable",
            "load_constraints": _load_constraints_for_backend(settings.engine.backend),
            "load_recommendations": _load_recommendations_for_backend(settings.engine.backend),
            "load_override": {},
            "definition": _model_definition_payload(
                model_settings,
                resolved_backend=settings.engine.backend,
            ),
        }

    def _has_load_overrides(self, load_request: AdminLoadRequest | None) -> bool:
        if load_request is None:
            return False
        fields_set = getattr(load_request, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(load_request, "__fields_set__", set())
        return bool(fields_set)
