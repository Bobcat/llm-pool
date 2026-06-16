from __future__ import annotations

from dataclasses import replace
import gc
from importlib import import_module
import threading
import time

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import AdminLoadRequest
from app.schemas import EngineResult
from app.schemas import ResponseMetrics
from app.schemas import ResponseRequest

from .common import LOGGER
from .common import _empty_cuda_allocator_cache
from .common import _estimate_model_artifact_size_mib
from .common import _exception_message
from .common import _load_constraints_for_backend
from .common import _load_recommendations_for_backend
from .common import _model_definition_payload
from .common import _model_supports_multi_turn
from .common import _model_thinking_modes
from .common import _normalize_gguf_cache_type_name
from .common import _normalize_gguf_flash_attn_mode
from .common import _query_gpu_memory
from .common import _query_primary_gpu_used_mib
from .common import _resolve_gguf_cache_type_constant
from .common import ModelRuntimeState
from .common import ModelStateError
from .common import RequestAdmissionError
from .common import UnknownModelError
from .scheduler import ExecutorSnapshot
from .scheduler import ReplicaRegistration
from .scheduler import RuntimeScheduler


class ModelRouterEngine:
    def __init__(self, settings: AppSettings) -> None:
        self._configured_models = dict(settings.engine.models)
        self._models: dict[str, object] = {}
        self._model_engines: dict[str, object] = {}
        self._loaded_replica_ids: dict[str, list[str]] = {}
        self._model_states: dict[str, ModelRuntimeState] = {}
        self._scheduler = RuntimeScheduler()
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)
        if not self._configured_models:
            raise ValueError("no configured models")

        for model_name, model_settings in settings.engine.models.items():
            backend = self._resolve_model_backend(settings.engine.backend, model_settings)
            self._model_states[model_name] = ModelRuntimeState(
                resolved_backend=backend,
                configured_enabled=model_settings.enabled,
            )

        for model_name, model_settings in self._configured_models.items():
            if not model_settings.enabled:
                continue
            try:
                self.load_model(model_name, settings)
            except Exception:
                LOGGER.exception("Failed to initialize model '%s'.", model_name)

    def list_models_payload(self) -> dict[str, object]:
        with self._state_lock:
            models = sorted(
                model_name
                for model_name, state in self._model_states.items()
                if state.lifecycle == "loaded"
            )
        return {"models": models}

    def complete(self, request: ResponseRequest) -> EngineResult:
        started_at = time.perf_counter()
        with self._state_lock:
            if request.model not in self._configured_models:
                raise UnknownModelError(request.model)
            model_settings = self._configured_models[request.model]
            state = self._model_states[request.model]
            if state.lifecycle != "loaded":
                raise ModelStateError(request.model, self._lifecycle_error_code(state.lifecycle))
            if state.resolved_backend == "openai_compatible" and not request.allow_remote:
                raise RequestAdmissionError(
                    code="remote_execution_disallowed",
                    status_code=403,
                    message="remote execution is not allowed for this request",
                )
            thinking_modes = _model_thinking_modes(
                state.resolved_backend,
                model_settings.prompt_format,
                model_settings.remote_thinking,
            )
            if request.thinking != "default" and request.thinking not in thinking_modes:
                raise RequestAdmissionError(
                    code="thinking_unsupported",
                    status_code=400,
                    message=f"model {request.model!r} does not support request-level thinking",
                )
            executor = self._scheduler.get(request.model)
            if executor is None:
                raise ModelStateError(request.model, "model_not_loaded")
            state.inflight_requests += 1
        try:
            result_future = executor.enqueue(request)
            result = result_future.result()
            total_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
            metrics_payload = (
                result.metrics.model_dump()
                if hasattr(result.metrics, "model_dump")
                else result.metrics.dict()
            )
            metrics_payload["engine_total_wall_ms"] = total_ms
            metrics_payload["pool_total_wall_ms"] = total_ms
            backend_wall_ms = metrics_payload.get("backend_inference_wall_ms")
            if backend_wall_ms is not None:
                metrics_payload["engine_outside_backend_wall_ms"] = max(0.0, total_ms - float(backend_wall_ms))
            return EngineResult(
                text=result.text,
                metrics=ResponseMetrics(**metrics_payload),
            )
        finally:
            with self._state_lock:
                state = self._model_states.get(request.model)
                if state is not None and state.inflight_requests > 0:
                    state.inflight_requests -= 1
                    self._state_changed.notify_all()

    def admin_models_payload(self, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        with self._state_lock:
            models: list[dict[str, object]] = []
            for model_name, model_settings in self._configured_models.items():
                models.append(self._admin_model_entry_locked(model_name, model_settings))
            return {"models": models}

    def admin_gpu_memory_payload(self, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        gpus, error = _query_gpu_memory()
        with self._state_lock:
            models: list[dict[str, object]] = []
            for model_name, model_settings in self._configured_models.items():
                models.append(self._admin_gpu_model_entry_locked(model_name, model_settings))
        return {"gpus": gpus, "models": models, "error": error}

    def load_model(
        self,
        model_name: str,
        settings: AppSettings | None = None,
        load_request: AdminLoadRequest | None = None,
    ) -> dict[str, object]:
        if settings is None:
            raise RuntimeError("settings are required to load a model")
        load_override = self._load_override_payload(load_request)
        requested_replica_count = self._replica_count_from_load_request(load_request)

        with self._state_lock:
            model_settings = self._configured_models.get(model_name)
            if model_settings is None:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle == "unloading":
                raise ModelStateError(model_name, "model_unloading")
            if state.lifecycle in {"loaded", "loading"}:
                if load_override or requested_replica_count is not None:
                    raise ValueError(
                        "replica count and load overrides can only be applied while the model is unloaded or failed; unload first"
                    )
                return self._admin_model_entry_locked(model_name, model_settings)
            replica_count = self._resolve_replica_count(model_name, model_settings, requested_replica_count)
            state.lifecycle = "loading"
            state.last_error = None
            resolved_backend = state.resolved_backend
            scoped_model_settings = self._apply_load_override(
                model_settings,
                resolved_backend=resolved_backend,
                load_override=load_override,
            )
            state.load_override = dict(load_override)
        gpu_used_before_mib = _query_primary_gpu_used_mib()
        replica_ids = self._replica_ids_for_model(model_name, replica_count)
        scoped_settings = replace(
            settings,
            engine=replace(
                settings.engine,
                backend=resolved_backend,
                models={replica_id: scoped_model_settings for replica_id in replica_ids},
            ),
        )

        try:
            backend_engine = self._build_backend_engine(resolved_backend, scoped_settings)
        except Exception as exc:
            message = _exception_message(exc)
            with self._state_lock:
                state = self._model_states[model_name]
                state.lifecycle = "failed"
                state.last_error = message
            LOGGER.exception(
                "Failed to load model '%s' from %s.",
                model_name,
                model_settings.model_path,
            )
            raise RuntimeError(message) from exc

        loaded_models = getattr(backend_engine, "_models", {})
        load_errors = getattr(backend_engine, "_load_errors", {})
        loaded_replicas = {
            replica_id: loaded_models[replica_id]
            for replica_id in replica_ids
            if replica_id in loaded_models
        }
        if len(loaded_replicas) != len(replica_ids):
            for runtime in loaded_replicas.values():
                self._cleanup_runtime(runtime)
            missing_replica_ids = [replica_id for replica_id in replica_ids if replica_id not in loaded_replicas]
            message = self._replica_load_error_message(missing_replica_ids, load_errors)
            with self._state_lock:
                state = self._model_states[model_name]
                state.lifecycle = "failed"
                state.last_error = message
            raise RuntimeError(message)

        gpu_used_after_mib = _query_primary_gpu_used_mib()
        observed_vram_mib: int | None = None
        if (
            gpu_used_before_mib is not None
            and gpu_used_after_mib is not None
            and gpu_used_after_mib >= gpu_used_before_mib
        ):
            delta = gpu_used_after_mib - gpu_used_before_mib
            if delta > 0:
                observed_vram_mib = delta

        with self._state_lock:
            existing_engine = self._find_backend_engine_locked(resolved_backend)
            target_engine = existing_engine or backend_engine
            if existing_engine is not None:
                for replica_id, runtime in loaded_replicas.items():
                    existing_engine._models[replica_id] = runtime
                if hasattr(existing_engine, "_load_errors"):
                    for replica_id in replica_ids:
                        existing_engine._load_errors.pop(replica_id, None)
            for replica_id, runtime in loaded_replicas.items():
                self._models[replica_id] = runtime
                self._model_engines[replica_id] = target_engine
            self._loaded_replica_ids[model_name] = list(replica_ids)
            self._register_executor_group(
                model_name=model_name,
                backend_engine=target_engine,
                replica_ids=replica_ids,
            )
            state = self._model_states[model_name]
            state.lifecycle = "loaded"
            state.last_error = None
            state.load_override = dict(load_override)
            if observed_vram_mib is not None:
                state.observed_vram_mib = observed_vram_mib
                state.observed_vram_replicas = replica_count
            return self._admin_model_entry_locked(model_name, model_settings)

    def unload_model(self, model_name: str, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
        executor = None
        with self._state_lock:
            model_settings = self._configured_models.get(model_name)
            if model_settings is None:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle == "loading":
                raise ModelStateError(model_name, "model_loading")
            if state.lifecycle in {"unloaded", "failed", "unloading"}:
                return self._admin_model_entry_locked(model_name, model_settings)

            state.lifecycle = "unloading"
            executor = self._scheduler.get(model_name)
        if executor is not None:
            executor.begin_shutdown()
        with self._state_lock:
            while state.inflight_requests > 0:
                self._state_changed.wait()

            replica_ids = list(self._loaded_replica_ids.pop(model_name, []))
            runtimes = []
            for replica_id in replica_ids:
                runtimes.append(self._models.pop(replica_id, None))
                backend_engine = self._model_engines.pop(replica_id, None)
                if backend_engine is not None:
                    backend_engine._models.pop(replica_id, None)
                    if hasattr(backend_engine, "_load_errors"):
                        backend_engine._load_errors.pop(replica_id, None)
            executor = self._scheduler.unregister(model_name)

        if executor is not None:
            executor.join(timeout=1.0)
        for runtime in runtimes:
            self._cleanup_runtime(runtime)

        with self._state_lock:
            state = self._model_states[model_name]
            state.lifecycle = "unloaded"
            state.last_error = None
            state.load_override = {}
            self._state_changed.notify_all()
            return self._admin_model_entry_locked(model_name, model_settings)

    def _admin_model_entry_locked(self, model_name: str, model_settings: ModelSettings) -> dict[str, object]:
        state = self._model_states[model_name]
        executor_snapshot = self._executor_snapshot_locked(model_name)
        (
            vram_estimate_mib,
            vram_estimate_replica_count,
            vram_estimate_source,
        ) = self._vram_estimate_locked(model_name, model_settings)
        current_replica_count = self._current_replica_count_locked(model_name, model_settings)
        return {
            "name": model_name,
            "resolved_backend": state.resolved_backend,
            "configured_enabled": state.configured_enabled,
            "runtime_state": state.lifecycle,
            "is_loaded": state.lifecycle == "loaded",
            "replicas": current_replica_count,
            "replica_max": model_settings.replica_max,
            "loaded_replicas": executor_snapshot.loaded_replicas,
            "inflight_requests": state.inflight_requests,
            "queue_depth": executor_snapshot.queue_depth,
            "runtime_inflight": executor_snapshot.runtime_inflight,
            "configured_target_inflight": executor_snapshot.configured_target_inflight,
            "effective_target_inflight": executor_snapshot.effective_target_inflight,
            "last_error": state.last_error,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_replica_count": vram_estimate_replica_count,
            "vram_estimate_source": vram_estimate_source,
            "load_constraints": _load_constraints_for_backend(state.resolved_backend),
            "load_recommendations": _load_recommendations_for_backend(state.resolved_backend),
            "load_override": dict(state.load_override),
            "capabilities": {
                "modalities": list(model_settings.modalities),
                "multi_turn": _model_supports_multi_turn(
                    state.resolved_backend,
                    model_settings.prompt_format,
                ),
                "thinking_modes": _model_thinking_modes(
                    state.resolved_backend,
                    model_settings.prompt_format,
                    model_settings.remote_thinking,
                ),
            },
            "definition": _model_definition_payload(
                model_settings,
                resolved_backend=state.resolved_backend,
            ),
        }

    def _admin_gpu_model_entry_locked(self, model_name: str, model_settings: ModelSettings) -> dict[str, object]:
        state = self._model_states[model_name]
        executor_snapshot = self._executor_snapshot_locked(model_name)
        (
            vram_estimate_mib,
            vram_estimate_replica_count,
            vram_estimate_source,
        ) = self._vram_estimate_locked(model_name, model_settings)
        return {
            "name": model_name,
            "runtime_state": state.lifecycle,
            "is_loaded": state.lifecycle == "loaded",
            "configured_target_inflight": executor_snapshot.configured_target_inflight,
            "effective_target_inflight": executor_snapshot.effective_target_inflight,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_replica_count": vram_estimate_replica_count,
            "vram_estimate_source": vram_estimate_source,
        }

    def _vram_estimate_locked(
        self,
        model_name: str,
        model_settings: ModelSettings,
    ) -> tuple[int | None, int | None, str]:
        state = self._model_states[model_name]
        if state.observed_vram_mib is not None:
            return (
                state.observed_vram_mib,
                state.observed_vram_replicas,
                "observed_load_delta",
            )
        if state.artifact_size_mib is None:
            state.artifact_size_mib = _estimate_model_artifact_size_mib(model_settings.model_path)
        if state.artifact_size_mib is not None:
            return state.artifact_size_mib, 1, "model_artifact_size"
        return None, None, "unavailable"

    def _find_backend_engine_locked(self, backend: str):
        for replica_id, engine in self._model_engines.items():
            public_model_name = self._public_model_from_replica_id(replica_id)
            state = self._model_states.get(public_model_name)
            if state is not None and state.resolved_backend == backend:
                return engine
        return None

    def _executor_snapshot_locked(self, model_name: str) -> ExecutorSnapshot:
        executor = self._scheduler.get(model_name)
        configured_target_inflight = self._configured_models[model_name].target_inflight
        if executor is None:
            return ExecutorSnapshot(
                queue_depth=0,
                runtime_inflight=0,
                configured_target_inflight=configured_target_inflight,
                effective_target_inflight=min(configured_target_inflight, 1),
                accepting_new_requests=False,
                loaded_replicas=0,
            )
        return executor.snapshot()

    def _register_executor_group(self, *, model_name: str, backend_engine: object, replica_ids: list[str]) -> None:
        self._scheduler.register(
            model_name=model_name,
            replicas=[
                ReplicaRegistration(
                    replica_id=replica_id,
                    complete_fn=self._runtime_complete_fn(backend_engine, replica_id),
                    runtime_capability=self._runtime_capability_for_model(model_name, backend_engine),
                )
                for replica_id in replica_ids
            ],
            configured_target_inflight=self._configured_models[model_name].target_inflight,
        )

    def _runtime_complete_fn(self, backend_engine: object, replica_id: str):
        def _complete(request: ResponseRequest) -> EngineResult:
            return backend_engine.complete(self._request_for_replica(request, replica_id))

        return _complete

    def _runtime_capability_for_model(self, model_name: str, backend_engine: object) -> int:
        del backend_engine
        state = self._model_states[model_name]
        if state.resolved_backend == "openai_compatible":
            return self._configured_models[model_name].target_inflight
        return 1

    def _cleanup_runtime(self, runtime: object | None) -> None:
        if runtime is None:
            _empty_cuda_allocator_cache()
            gc.collect()
            return

        close_runtime = getattr(runtime, "close", None)
        if callable(close_runtime):
            try:
                close_runtime()
            except Exception:
                LOGGER.warning("Failed to close runtime during cleanup.", exc_info=True)

        generator = getattr(runtime, "generator", None)
        clear_queue = getattr(generator, "clear_queue", None)
        if callable(clear_queue):
            try:
                clear_queue()
            except Exception:
                LOGGER.warning("Failed to clear backend queue during runtime cleanup.", exc_info=True)

        llm = getattr(runtime, "llm", None)
        close = getattr(llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.warning("Failed to close GGUF runtime during cleanup.", exc_info=True)

        model = getattr(runtime, "model", None)
        cache = getattr(runtime, "cache", None)

        unload_model = getattr(model, "unload", None)
        if callable(unload_model):
            try:
                unload_model()
            except Exception:
                LOGGER.warning("Failed to unload ExLlamaV3 model during cleanup.", exc_info=True)

        detach_cache = getattr(cache, "detach_from_model", None)
        if callable(detach_cache):
            try:
                if model is not None:
                    detach_cache(model)
                else:
                    detach_cache()
            except Exception:
                LOGGER.warning("Failed to detach ExLlamaV3 cache during cleanup.", exc_info=True)

        if model is not None or cache is not None or generator is not None:
            try:
                from exllamav3.util.tensor import g_tensor_cache  # type: ignore

                g_tensor_cache.drop_all()
            except Exception:
                pass
            try:
                from exllamav3.util.memory import free_mem as exllama_free_mem  # type: ignore

                exllama_free_mem()
            except Exception:
                pass

        for attr_name in (
            "generator",
            "llm",
            "cache",
            "model",
            "tokenizer",
            "job_class",
            "sampler_class",
        ):
            if hasattr(runtime, attr_name):
                try:
                    setattr(runtime, attr_name, None)
                except Exception:
                    pass

        _empty_cuda_allocator_cache()
        gc.collect()

    def _lifecycle_error_code(self, lifecycle: str) -> str:
        if lifecycle == "loading":
            return "model_loading"
        if lifecycle == "unloading":
            return "model_unloading"
        if lifecycle == "failed":
            return "model_failed"
        return "model_not_loaded"

    def _resolve_model_backend(self, default_backend: str, model_settings: ModelSettings) -> str:
        backend = model_settings.backend or default_backend
        backend = backend.strip().lower()
        if backend == "":
            raise ValueError("backend cannot be empty")
        return backend

    def _replica_count_from_load_request(self, load_request: AdminLoadRequest | None) -> int | None:
        if load_request is None:
            return None
        fields_set = getattr(load_request, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(load_request, "__fields_set__", set())
        if "replicas" not in fields_set:
            return None
        return load_request.replicas

    def _resolve_replica_count(
        self,
        model_name: str,
        model_settings: ModelSettings,
        requested_replica_count: int | None,
    ) -> int:
        replica_count = model_settings.replicas if requested_replica_count is None else int(requested_replica_count)
        if replica_count <= 0:
            raise ValueError("replica count must be a positive integer")
        if replica_count > model_settings.replica_max:
            raise ValueError(
                f"replica count {replica_count} exceeds replica_max {model_settings.replica_max} for model '{model_name}'"
            )
        return replica_count

    def _replica_ids_for_model(self, model_name: str, replica_count: int) -> list[str]:
        return [f"{model_name}#{index}" for index in range(1, max(1, int(replica_count)) + 1)]

    def _public_model_from_replica_id(self, replica_id: str) -> str:
        if "#" not in replica_id:
            return replica_id
        return replica_id.rsplit("#", 1)[0]

    def _replica_load_error_message(
        self,
        missing_replica_ids: list[str],
        load_errors: dict[str, object],
    ) -> str:
        detailed_errors = [
            f"{replica_id}: {load_errors[replica_id]}"
            for replica_id in missing_replica_ids
            if replica_id in load_errors
        ]
        if detailed_errors:
            return "; ".join(str(item) for item in detailed_errors)
        return "replica group failed to load completely"

    def _current_replica_count_locked(self, model_name: str, model_settings: ModelSettings) -> int:
        replica_ids = self._loaded_replica_ids.get(model_name, [])
        if replica_ids:
            return len(replica_ids)
        return model_settings.replicas

    def _request_for_replica(self, request: ResponseRequest, replica_id: str) -> ResponseRequest:
        if hasattr(request, "model_copy"):
            return request.model_copy(update={"model": replica_id})
        return request.copy(update={"model": replica_id})

    def _load_override_payload(self, load_request: AdminLoadRequest | None) -> dict[str, object | None]:
        if load_request is None:
            return {}
        fields_set = getattr(load_request, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(load_request, "__fields_set__", set())
        payload: dict[str, object | None] = {}
        for field_name in fields_set:
            if field_name == "replicas":
                continue
            payload[field_name] = getattr(load_request, field_name)
        has_exllama_k_bits = "exllama_cache_k_bits" in payload
        has_exllama_v_bits = "exllama_cache_v_bits" in payload
        if has_exllama_k_bits or has_exllama_v_bits:
            if "exllama_cache_quant" in payload:
                raise ValueError(
                    "exllama_cache_quant cannot be combined with exllama_cache_k_bits/exllama_cache_v_bits"
                )
            if not (has_exllama_k_bits and has_exllama_v_bits):
                raise ValueError("exllama_cache_k_bits and exllama_cache_v_bits must be provided together")
            k_bits = payload.pop("exllama_cache_k_bits")
            v_bits = payload.pop("exllama_cache_v_bits")
            if k_bits is None and v_bits is None:
                payload["exllama_cache_quant"] = None
            elif isinstance(k_bits, int) and isinstance(v_bits, int):
                payload["exllama_cache_quant"] = f"{k_bits},{v_bits}"
            else:
                raise ValueError("exllama_cache_k_bits and exllama_cache_v_bits must both be integers or both be null")
        if "gguf_flash_attn" in payload:
            gguf_flash_attn = payload["gguf_flash_attn"]
            if gguf_flash_attn is None or not isinstance(gguf_flash_attn, str):
                raise ValueError("gguf_flash_attn load override must be one of: on, off, auto")
            payload["gguf_flash_attn"] = _normalize_gguf_flash_attn_mode(gguf_flash_attn)
        return payload

    def _apply_load_override(
        self,
        model_settings: ModelSettings,
        *,
        resolved_backend: str,
        load_override: dict[str, object | None],
    ) -> ModelSettings:
        if not load_override:
            return replace(model_settings, enabled=True)

        if resolved_backend == "gguf":
            unsupported = sorted(
                field_name
                for field_name in load_override
                if field_name not in {"gguf_n_ctx", "gguf_flash_attn", "gguf_type_k", "gguf_type_v"}
            )
            if unsupported:
                names = ", ".join(unsupported)
                raise ValueError(f"unsupported load override for gguf backend: {names}")

            replacement_kwargs: dict[str, object | None] = {"enabled": True}

            if "gguf_n_ctx" in load_override:
                gguf_n_ctx = load_override["gguf_n_ctx"]
                if not isinstance(gguf_n_ctx, int):
                    raise ValueError("gguf_n_ctx load override must be a positive integer")
                replacement_kwargs["gguf_n_ctx"] = gguf_n_ctx

            if "gguf_flash_attn" in load_override:
                gguf_flash_attn = load_override["gguf_flash_attn"]
                if not isinstance(gguf_flash_attn, str):
                    raise ValueError("gguf_flash_attn load override must be one of: on, off, auto")
                replacement_kwargs["gguf_flash_attn"] = _normalize_gguf_flash_attn_mode(gguf_flash_attn)

            if "gguf_type_k" in load_override:
                gguf_type_k = load_override["gguf_type_k"]
                if gguf_type_k is not None and not isinstance(gguf_type_k, str):
                    raise ValueError("gguf_type_k load override must be a string or null")
                if isinstance(gguf_type_k, str):
                    normalized_gguf_type_k = _normalize_gguf_cache_type_name(gguf_type_k)
                    self._validate_gguf_cache_type_name(normalized_gguf_type_k)
                    replacement_kwargs["gguf_type_k"] = normalized_gguf_type_k
                else:
                    replacement_kwargs["gguf_type_k"] = None

            if "gguf_type_v" in load_override:
                gguf_type_v = load_override["gguf_type_v"]
                if gguf_type_v is not None and not isinstance(gguf_type_v, str):
                    raise ValueError("gguf_type_v load override must be a string or null")
                if isinstance(gguf_type_v, str):
                    normalized_gguf_type_v = _normalize_gguf_cache_type_name(gguf_type_v)
                    self._validate_gguf_cache_type_name(normalized_gguf_type_v)
                    replacement_kwargs["gguf_type_v"] = normalized_gguf_type_v
                else:
                    replacement_kwargs["gguf_type_v"] = None

            return replace(model_settings, **replacement_kwargs)

        if resolved_backend == "exllamav3":
            unsupported = sorted(
                field_name
                for field_name in load_override
                if field_name not in {"exllama_cache_size", "exllama_cache_quant", "exllama_max_rq_tokens"}
            )
            if unsupported:
                names = ", ".join(unsupported)
                raise ValueError(f"unsupported load override for exllamav3 backend: {names}")

            replacement_kwargs: dict[str, object | None] = {"enabled": True}

            if "exllama_cache_size" in load_override:
                cache_size = load_override["exllama_cache_size"]
                if not isinstance(cache_size, int):
                    raise ValueError("exllama_cache_size load override must be a positive integer")
                if cache_size <= 0 or cache_size % 256 != 0:
                    raise ValueError("exllama_cache_size load override must be a positive multiple of 256")
                replacement_kwargs["exllama_cache_size"] = cache_size

            if "exllama_cache_quant" in load_override:
                cache_quant = load_override["exllama_cache_quant"]
                if cache_quant is not None and not isinstance(cache_quant, str):
                    raise ValueError("exllama_cache_quant load override must be a string or null")
                if isinstance(cache_quant, str):
                    self._parse_exllama_cache_quant(cache_quant)
                replacement_kwargs["exllama_cache_quant"] = cache_quant

            if "exllama_max_rq_tokens" in load_override:
                max_rq_tokens = load_override["exllama_max_rq_tokens"]
                if max_rq_tokens is not None and not isinstance(max_rq_tokens, int):
                    raise ValueError("exllama_max_rq_tokens load override must be a positive integer or null")
                replacement_kwargs["exllama_max_rq_tokens"] = max_rq_tokens

            return replace(model_settings, **replacement_kwargs)

        if resolved_backend == "vllm":
            unsupported = sorted(
                field_name
                for field_name in load_override
                if field_name
                not in {
                    "vllm_max_model_len",
                    "vllm_kv_cache_dtype",
                    "vllm_kv_cache_memory_bytes",
                    "vllm_max_pixels",
                }
            )
            if unsupported:
                names = ", ".join(unsupported)
                raise ValueError(f"unsupported load override for vllm backend: {names}")

            replacement_kwargs: dict[str, object | None] = {"enabled": True}

            if "vllm_max_model_len" in load_override:
                max_model_len = load_override["vllm_max_model_len"]
                if not isinstance(max_model_len, int) or max_model_len <= 0:
                    raise ValueError("vllm_max_model_len load override must be a positive integer")
                replacement_kwargs["vllm_max_model_len"] = max_model_len

            if "vllm_kv_cache_dtype" in load_override:
                kv_cache_dtype = load_override["vllm_kv_cache_dtype"]
                if not isinstance(kv_cache_dtype, str) or kv_cache_dtype.strip() == "":
                    raise ValueError("vllm_kv_cache_dtype load override must be a non-empty string")
                replacement_kwargs["vllm_kv_cache_dtype"] = kv_cache_dtype.strip()

            if "vllm_kv_cache_memory_bytes" in load_override:
                kv_cache_bytes = load_override["vllm_kv_cache_memory_bytes"]
                if not isinstance(kv_cache_bytes, int) or kv_cache_bytes <= 0:
                    raise ValueError("vllm_kv_cache_memory_bytes load override must be a positive integer")
                replacement_kwargs["vllm_kv_cache_memory_bytes"] = kv_cache_bytes

            if "vllm_max_pixels" in load_override:
                max_pixels = load_override["vllm_max_pixels"]
                if not isinstance(max_pixels, int) or max_pixels <= 0:
                    raise ValueError("vllm_max_pixels load override must be a positive integer")
                mm_kwargs = {key: value for key, value in model_settings.vllm_mm_processor_kwargs}
                mm_kwargs["max_pixels"] = max_pixels
                replacement_kwargs["vllm_mm_processor_kwargs"] = tuple(mm_kwargs.items())

            return replace(model_settings, **replacement_kwargs)

        if resolved_backend == "llama_server":
            unsupported = sorted(
                field_name
                for field_name in load_override
                if field_name
                not in {
                    "llama_server_n_ctx",
                    "llama_server_image_max_tokens",
                    "llama_server_spec_type",
                    "llama_server_spec_draft_n_max",
                    "llama_server_spec_draft_p_min",
                }
            )
            if unsupported:
                names = ", ".join(unsupported)
                raise ValueError(f"unsupported load override for llama_server backend: {names}")

            replacement_kwargs: dict[str, object | None] = {"enabled": True}

            if "llama_server_n_ctx" in load_override:
                n_ctx = load_override["llama_server_n_ctx"]
                if n_ctx is not None and (not isinstance(n_ctx, int) or n_ctx <= 0):
                    raise ValueError("llama_server_n_ctx load override must be a positive integer or null")
                replacement_kwargs["llama_server_n_ctx"] = n_ctx

            if "llama_server_image_max_tokens" in load_override:
                image_max_tokens = load_override["llama_server_image_max_tokens"]
                if image_max_tokens is not None and (
                    not isinstance(image_max_tokens, int) or image_max_tokens <= 0
                ):
                    raise ValueError("llama_server_image_max_tokens load override must be a positive integer or null")
                replacement_kwargs["llama_server_image_max_tokens"] = image_max_tokens

            if "llama_server_spec_type" in load_override:
                spec_type = load_override["llama_server_spec_type"]
                if spec_type is not None:
                    if not isinstance(spec_type, str) or spec_type.strip() != "draft-mtp":
                        raise ValueError("llama_server_spec_type load override must be 'draft-mtp' or null")
                    spec_type = spec_type.strip()
                replacement_kwargs["llama_server_spec_type"] = spec_type

            if "llama_server_spec_draft_n_max" in load_override:
                draft_n_max = load_override["llama_server_spec_draft_n_max"]
                if draft_n_max is not None and (
                    not isinstance(draft_n_max, int) or draft_n_max < 1 or draft_n_max > 6
                ):
                    raise ValueError(
                        "llama_server_spec_draft_n_max load override must be an integer between 1 and 6 or null"
                    )
                replacement_kwargs["llama_server_spec_draft_n_max"] = draft_n_max

            if "llama_server_spec_draft_p_min" in load_override:
                draft_p_min = load_override["llama_server_spec_draft_p_min"]
                if draft_p_min is not None and (
                    not isinstance(draft_p_min, (float, int)) or draft_p_min < 0.0 or draft_p_min > 1.0
                ):
                    raise ValueError(
                        "llama_server_spec_draft_p_min load override must be between 0.0 and 1.0 or null"
                    )
                replacement_kwargs["llama_server_spec_draft_p_min"] = (
                    float(draft_p_min) if draft_p_min is not None else None
                )

            return replace(model_settings, **replacement_kwargs)

        raise ValueError(f"load overrides are not supported for backend: {resolved_backend!r}")

    def _build_backend_engine(self, backend: str, settings: AppSettings):
        engine_module = import_module("app.engine")
        if backend == "ct2":
            return engine_module.Ct2Engine(settings)
        if backend == "exllamav3":
            return engine_module.ExLlamaV3Engine(settings)
        if backend == "gguf":
            return engine_module.LlamaCppEngine(settings)
        if backend == "llama_server":
            return engine_module.LlamaServerEngine(settings)
        if backend == "openai_compatible":
            return engine_module.OpenAICompatibleEngine(settings)
        if backend == "vllm":
            return engine_module.VllmEngine(settings)
        if backend == "stub":
            return engine_module.StubEngine(settings)
        raise ValueError(f"unsupported engine backend: {backend!r}")

    def _parse_exllama_cache_quant(self, cache_quant: str) -> tuple[int, int]:
        split = [part.strip() for part in cache_quant.split(",") if part.strip() != ""]
        if len(split) == 1:
            bits = int(split[0])
            return bits, bits
        if len(split) == 2:
            return int(split[0]), int(split[1])
        raise ValueError("exllama_cache_quant load override must be '<bits>' or '<k_bits>,<v_bits>'")

    def _validate_gguf_cache_type_name(self, cache_type_name: str) -> None:
        try:
            import llama_cpp as llama_cpp_module
        except ImportError:
            return
        _resolve_gguf_cache_type_constant(cache_type_name, llama_cpp_module=llama_cpp_module)

    def close(self) -> None:
        self._scheduler.close()
        with self._state_lock:
            runtimes = list(self._models.values())
            self._models.clear()
            self._model_engines.clear()
            self._loaded_replica_ids.clear()
            for state in self._model_states.values():
                if state.lifecycle in {"loaded", "loading", "unloading"}:
                    state.lifecycle = "unloaded"
                    state.inflight_requests = 0
            self._state_changed.notify_all()
        for runtime in runtimes:
            self._cleanup_runtime(runtime)
