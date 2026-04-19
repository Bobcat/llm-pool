from __future__ import annotations

from dataclasses import replace
import gc
from importlib import import_module
import threading

from app.config import AppSettings
from app.config import ModelSettings
from app.schemas import AdminLoadRequest
from app.schemas import EngineResult
from app.schemas import ResponseRequest

from .common import LOGGER
from .common import _empty_cuda_allocator_cache
from .common import _estimate_model_artifact_size_mib
from .common import _exception_message
from .common import _load_constraints_for_backend
from .common import _load_recommendations_for_backend
from .common import _model_definition_payload
from .common import _normalize_gguf_cache_type_name
from .common import _normalize_gguf_flash_attn_mode
from .common import _query_gpu_memory
from .common import _query_primary_gpu_used_mib
from .common import _resolve_gguf_cache_type_constant
from .common import ModelRuntimeState
from .common import ModelStateError
from .common import UnknownModelError


class ModelRouterEngine:
    def __init__(self, settings: AppSettings) -> None:
        self._configured_models = dict(settings.engine.models)
        self._models: dict[str, object] = {}
        self._model_engines: dict[str, object] = {}
        self._model_states: dict[str, ModelRuntimeState] = {}
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)
        if not self._configured_models:
            raise ValueError("no configured models")
        grouped_models: dict[str, dict[str, ModelSettings]] = {}
        for model_name, model_settings in settings.engine.models.items():
            backend = self._resolve_model_backend(settings.engine.backend, model_settings)
            self._model_states[model_name] = ModelRuntimeState(
                resolved_backend=backend,
                configured_enabled=model_settings.enabled,
            )
            if not model_settings.enabled:
                continue
            grouped_models.setdefault(backend, {})[model_name] = model_settings

        for backend, models in grouped_models.items():
            scoped_settings = replace(
                settings,
                engine=replace(
                    settings.engine,
                    backend=backend,
                    models=models,
                ),
            )
            try:
                backend_engine = self._build_backend_engine(backend, scoped_settings)
            except Exception as exc:
                message = _exception_message(exc)
                for model_name in models:
                    state = self._model_states[model_name]
                    state.lifecycle = "failed"
                    state.last_error = message
                LOGGER.exception(
                    "Failed to initialize backend '%s'; skipping %d model(s).",
                    backend,
                    len(models),
                )
                continue
            loaded_models = getattr(backend_engine, "_models", {})
            load_errors = getattr(backend_engine, "_load_errors", {})
            for model_name in models:
                state = self._model_states[model_name]
                if model_name in loaded_models:
                    state.lifecycle = "loaded"
                    state.last_error = None
                else:
                    state.lifecycle = "failed"
                    state.last_error = str(load_errors.get(model_name, "model failed to load"))
            if not loaded_models:
                LOGGER.warning(
                    "Backend '%s' initialized without loaded models; skipping backend.",
                    backend,
                )
                continue
            for model_name, runtime in loaded_models.items():
                self._models[model_name] = runtime
                self._model_engines[model_name] = backend_engine

    def complete(self, request: ResponseRequest) -> EngineResult:
        with self._state_lock:
            if request.model not in self._configured_models:
                raise UnknownModelError(request.model)
            state = self._model_states[request.model]
            if state.lifecycle != "loaded":
                raise ModelStateError(request.model, self._lifecycle_error_code(state.lifecycle))
            engine = self._model_engines.get(request.model)
            if engine is None:
                raise ModelStateError(request.model, "model_not_loaded")
            state.inflight_requests += 1
        try:
            return engine.complete(request)
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

        with self._state_lock:
            model_settings = self._configured_models.get(model_name)
            if model_settings is None:
                raise UnknownModelError(model_name)
            state = self._model_states[model_name]
            if state.lifecycle == "unloading":
                raise ModelStateError(model_name, "model_unloading")
            if state.lifecycle in {"loaded", "loading"}:
                if load_override:
                    raise ValueError(
                        "load overrides can only be applied while the model is unloaded or failed; unload first"
                    )
                return self._admin_model_entry_locked(model_name, model_settings)
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

        scoped_settings = replace(
            settings,
            engine=replace(
                settings.engine,
                backend=resolved_backend,
                models={model_name: scoped_model_settings},
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
        runtime = loaded_models.get(model_name)
        if runtime is None:
            message = str(load_errors.get(model_name, "model failed to load"))
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
                existing_engine._models[model_name] = runtime
                if hasattr(existing_engine, "_load_errors"):
                    existing_engine._load_errors.pop(model_name, None)
            self._models[model_name] = runtime
            self._model_engines[model_name] = target_engine
            state = self._model_states[model_name]
            state.lifecycle = "loaded"
            state.last_error = None
            state.load_override = dict(load_override)
            if observed_vram_mib is not None:
                state.observed_vram_mib = observed_vram_mib
            return self._admin_model_entry_locked(model_name, model_settings)

    def unload_model(self, model_name: str, settings: AppSettings | None = None) -> dict[str, object]:
        del settings
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
            while state.inflight_requests > 0:
                self._state_changed.wait()

            runtime = self._models.pop(model_name, None)
            backend_engine = self._model_engines.pop(model_name, None)
            if backend_engine is not None:
                backend_engine._models.pop(model_name, None)
                if hasattr(backend_engine, "_load_errors"):
                    backend_engine._load_errors.pop(model_name, None)

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
        vram_estimate_mib, vram_estimate_source = self._vram_estimate_locked(model_name, model_settings)
        return {
            "name": model_name,
            "resolved_backend": state.resolved_backend,
            "configured_enabled": state.configured_enabled,
            "runtime_state": state.lifecycle,
            "is_loaded": state.lifecycle == "loaded",
            "inflight_requests": state.inflight_requests,
            "last_error": state.last_error,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": vram_estimate_source,
            "load_constraints": _load_constraints_for_backend(state.resolved_backend),
            "load_recommendations": _load_recommendations_for_backend(state.resolved_backend),
            "load_override": dict(state.load_override),
            "definition": _model_definition_payload(
                model_settings,
                resolved_backend=state.resolved_backend,
            ),
        }

    def _admin_gpu_model_entry_locked(self, model_name: str, model_settings: ModelSettings) -> dict[str, object]:
        state = self._model_states[model_name]
        vram_estimate_mib, vram_estimate_source = self._vram_estimate_locked(model_name, model_settings)
        return {
            "name": model_name,
            "runtime_state": state.lifecycle,
            "is_loaded": state.lifecycle == "loaded",
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": vram_estimate_source,
        }

    def _vram_estimate_locked(
        self,
        model_name: str,
        model_settings: ModelSettings,
    ) -> tuple[int | None, str]:
        state = self._model_states[model_name]
        if state.observed_vram_mib is not None:
            return state.observed_vram_mib, "observed_load_delta"
        if state.artifact_size_mib is None:
            state.artifact_size_mib = _estimate_model_artifact_size_mib(model_settings.model_path)
        if state.artifact_size_mib is not None:
            return state.artifact_size_mib, "model_artifact_size"
        return None, "unavailable"

    def _find_backend_engine_locked(self, backend: str):
        for loaded_model_name, engine in self._model_engines.items():
            state = self._model_states.get(loaded_model_name)
            if state is not None and state.resolved_backend == backend:
                return engine
        return None

    def _cleanup_runtime(self, runtime: object | None) -> None:
        if runtime is None:
            _empty_cuda_allocator_cache()
            gc.collect()
            return

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

    def _load_override_payload(self, load_request: AdminLoadRequest | None) -> dict[str, object | None]:
        if load_request is None:
            return {}
        fields_set = getattr(load_request, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(load_request, "__fields_set__", set())
        payload: dict[str, object | None] = {}
        for field_name in fields_set:
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

        raise ValueError(f"load overrides are not supported for backend: {resolved_backend!r}")

    def _build_backend_engine(self, backend: str, settings: AppSettings):
        engine_module = import_module("app.engine")
        if backend == "ct2":
            return engine_module.Ct2Engine(settings)
        if backend == "exllamav3":
            return engine_module.ExLlamaV3Engine(settings)
        if backend == "gguf":
            return engine_module.LlamaCppEngine(settings)
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
