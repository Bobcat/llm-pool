from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
import unittest
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import AppSettings
    from app.config import EngineSettings
    from app.config import ModelSettings
    from app.config import ServiceSettings
    import app.engine as engine_module
    import app.engine.router as router_module
    from app.engine import ModelRouterEngine
    from app.engine import build_engine
    from app.schemas import AdminLoadRequest
    from app.schemas import EngineResult
    from app.schemas import ResponseRequest


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class ModelRouterEngineTests(unittest.TestCase):
    def test_dispatches_by_model_backend(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "ct2-model": ModelSettings(model_path="/models/ct2"),
                    "exl-model": ModelSettings(model_path="/models/exl3", backend="exllamav3"),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        class FakeExLlamaV3Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"exl3:{request.model}")

        with (
            mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine),
            mock.patch.object(engine_module, "ExLlamaV3Engine", FakeExLlamaV3Engine),
        ):
            engine = ModelRouterEngine(settings)
            ct2_result = engine.complete(ResponseRequest(model="ct2-model", input="hello"))
            exl_result = engine.complete(ResponseRequest(model="exl-model", input="hello"))

        self.assertEqual(ct2_result.text, "ct2:ct2-model")
        self.assertEqual(exl_result.text, "exl3:exl-model")
        self.assertEqual(sorted(engine._models.keys()), ["ct2-model", "exl-model"])

    def test_dispatches_gguf_backend(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "ct2-model": ModelSettings(model_path="/models/ct2"),
                    "gguf-model": ModelSettings(
                        model_path="/models/test.gguf",
                        backend="gguf",
                    ),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        class FakeLlamaCppEngine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"gguf:{request.model}")

        with (
            mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine),
            mock.patch.object(engine_module, "LlamaCppEngine", FakeLlamaCppEngine),
        ):
            engine = ModelRouterEngine(settings)
            gguf_result = engine.complete(ResponseRequest(model="gguf-model", input="hello"))

        self.assertEqual(gguf_result.text, "gguf:gguf-model")

    def test_build_engine_uses_model_router_for_non_stub_backends(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "ct2-model": ModelSettings(model_path="/models/ct2"),
                },
            ),
        )

        with mock.patch.object(engine_module.ModelRouterEngine, "__init__", return_value=None) as router_init:
            engine = build_engine(settings)

        self.assertIsInstance(engine, ModelRouterEngine)
        router_init.assert_called_once_with(settings)

    def test_admin_models_payload_reports_loaded_failed_and_unloaded(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "ct2-model": ModelSettings(model_path="/models/ct2"),
                    "broken-model": ModelSettings(model_path="/models/broken"),
                    "disabled-model": ModelSettings(model_path="/models/disabled", enabled=False),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {"ct2-model": object()}
                self._load_errors = {"broken-model": "boom"}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        payload = engine.admin_models_payload()

        self.assertEqual(len(payload["models"]), 3)

        loaded_model = payload["models"][0]
        self.assertEqual(loaded_model["name"], "ct2-model")
        self.assertEqual(loaded_model["runtime_state"], "loaded")
        self.assertTrue(loaded_model["is_loaded"])
        self.assertIsNone(loaded_model["last_error"])
        self.assertIn("vram_estimate_mib", loaded_model)
        self.assertIn("vram_estimate_source", loaded_model)
        self.assertEqual(loaded_model["load_constraints"], {})
        self.assertIn("device", loaded_model["definition"])
        self.assertIn("compute_type", loaded_model["definition"])
        self.assertNotIn("exllama_cache_size", loaded_model["definition"])
        self.assertNotIn("gguf_n_ctx", loaded_model["definition"])

        failed_model = payload["models"][1]
        self.assertEqual(failed_model["name"], "broken-model")
        self.assertEqual(failed_model["runtime_state"], "failed")
        self.assertFalse(failed_model["is_loaded"])
        self.assertEqual(failed_model["last_error"], "boom")
        self.assertIn("vram_estimate_mib", failed_model)
        self.assertIn("vram_estimate_source", failed_model)
        self.assertEqual(failed_model["load_constraints"], {})
        self.assertIn("device", failed_model["definition"])
        self.assertIn("compute_type", failed_model["definition"])
        self.assertNotIn("exllama_cache_size", failed_model["definition"])
        self.assertNotIn("gguf_n_ctx", failed_model["definition"])

        unloaded_model = payload["models"][2]
        self.assertEqual(unloaded_model["name"], "disabled-model")
        self.assertEqual(unloaded_model["runtime_state"], "unloaded")
        self.assertFalse(unloaded_model["is_loaded"])
        self.assertIsNone(unloaded_model["last_error"])
        self.assertIn("vram_estimate_mib", unloaded_model)
        self.assertIn("vram_estimate_source", unloaded_model)
        self.assertEqual(unloaded_model["load_constraints"], {})
        self.assertIn("device", unloaded_model["definition"])
        self.assertIn("compute_type", unloaded_model["definition"])
        self.assertNotIn("exllama_cache_size", unloaded_model["definition"])
        self.assertNotIn("gguf_n_ctx", unloaded_model["definition"])

    def test_admin_models_payload_reports_gguf_load_constraints(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="gguf",
                models={
                    "gguf-model": ModelSettings(model_path="/models/test.gguf", backend="gguf"),
                },
            ),
        )

        class FakeLlamaCppEngine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"gguf:{request.model}")

        with mock.patch.object(engine_module, "LlamaCppEngine", FakeLlamaCppEngine):
            engine = ModelRouterEngine(settings)

        payload = engine.admin_models_payload()
        self.assertEqual(
            payload["models"][0]["load_constraints"],
            {
                "gguf_n_ctx": {
                    "kind": "integer",
                    "minimum": 1,
                    "step": 1,
                },
                "gguf_type_k": {
                    "kind": "string_or_null",
                    "format": "ggml_type_name",
                    "examples": ["f16", "q8_0", "q4_0"],
                },
                "gguf_type_v": {
                    "kind": "string_or_null",
                    "format": "ggml_type_name",
                    "examples": ["f16", "q8_0", "q4_0"],
                },
            },
        )
        self.assertIn("gguf_n_ctx", payload["models"][0]["definition"])
        self.assertIn("gguf_flash_attn", payload["models"][0]["definition"])
        self.assertIn("gguf_type_k", payload["models"][0]["definition"])
        self.assertIn("gguf_type_v", payload["models"][0]["definition"])
        self.assertNotIn("device", payload["models"][0]["definition"])
        self.assertNotIn("compute_type", payload["models"][0]["definition"])
        self.assertNotIn("exllama_cache_size", payload["models"][0]["definition"])

    def test_admin_models_payload_reports_exllamav3_load_constraints(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="exllamav3",
                models={
                    "exl-model": ModelSettings(model_path="/models/exl3", backend="exllamav3"),
                },
            ),
        )

        class FakeExLlamaV3Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"exllamav3:{request.model}")

        with mock.patch.object(engine_module, "ExLlamaV3Engine", FakeExLlamaV3Engine):
            engine = ModelRouterEngine(settings)

        payload = engine.admin_models_payload()
        self.assertEqual(
            payload["models"][0]["load_constraints"],
            {
                "exllama_cache_size": {
                    "kind": "integer",
                    "minimum": 256,
                    "step": 256,
                },
                "exllama_max_rq_tokens": {
                    "kind": "integer",
                    "minimum": 1,
                    "step": 1,
                },
                "exllama_cache_quant": {
                    "kind": "string_or_null",
                    "format": "<bits>|<k_bits>,<v_bits>",
                },
            },
        )
        self.assertIn("device", payload["models"][0]["definition"])
        self.assertIn("exllama_cache_size", payload["models"][0]["definition"])
        self.assertIn("exllama_max_rq_tokens", payload["models"][0]["definition"])
        self.assertNotIn("compute_type", payload["models"][0]["definition"])
        self.assertNotIn("gguf_n_ctx", payload["models"][0]["definition"])

    def test_tracks_inflight_requests_around_complete(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "ct2-model": ModelSettings(model_path="/models/ct2"),
                },
            ),
        )
        gate = threading.Event()
        entered = threading.Event()

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {"ct2-model": object()}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                entered.set()
                gate.wait(timeout=1.0)
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        result_holder: dict[str, EngineResult] = {}

        def run_complete() -> None:
            result_holder["result"] = engine.complete(ResponseRequest(model="ct2-model", input="hello"))

        thread = threading.Thread(target=run_complete)
        thread.start()
        entered.wait(timeout=1.0)

        payload = engine.admin_models_payload()
        self.assertEqual(payload["models"][0]["inflight_requests"], 1)

        gate.set()
        thread.join(timeout=1.0)

        self.assertEqual(result_holder["result"].text, "ct2:ct2-model")
        payload = engine.admin_models_payload()
        self.assertEqual(payload["models"][0]["inflight_requests"], 0)

    def test_load_model_can_load_disabled_model(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "enabled-model": ModelSettings(model_path="/models/enabled"),
                    "disabled-model": ModelSettings(model_path="/models/disabled", enabled=False),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)
            entry = engine.load_model("disabled-model", settings)

        self.assertEqual(entry["name"], "disabled-model")
        self.assertEqual(entry["runtime_state"], "loaded")
        self.assertTrue(entry["is_loaded"])
        self.assertIn("disabled-model", engine._models)
        self.assertEqual(engine.admin_models_payload()["models"][1]["runtime_state"], "loaded")

    def test_load_model_applies_gguf_load_overrides(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="gguf",
                models={
                    "gguf-model": ModelSettings(
                        model_path="/models/test.gguf",
                        backend="gguf",
                        enabled=False,
                        gguf_n_ctx=4096,
                        gguf_type_k="f16",
                        gguf_type_v="f16",
                    ),
                },
            ),
        )

        captured: dict[str, object] = {}

        class FakeBackend:
            def __init__(self):
                self._models = {"gguf-model": object()}
                self._load_errors = {}

        engine = ModelRouterEngine(settings)

        def fake_build_backend(backend: str, scoped_settings: AppSettings):
            captured["backend"] = backend
            captured["gguf_n_ctx"] = scoped_settings.engine.models["gguf-model"].gguf_n_ctx
            captured["gguf_type_k"] = scoped_settings.engine.models["gguf-model"].gguf_type_k
            captured["gguf_type_v"] = scoped_settings.engine.models["gguf-model"].gguf_type_v
            return FakeBackend()

        with mock.patch.object(engine, "_build_backend_engine", side_effect=fake_build_backend):
            entry = engine.load_model(
                "gguf-model",
                settings,
                AdminLoadRequest(
                    gguf_n_ctx=32768,
                    gguf_type_k="Q8_0",
                    gguf_type_v="q4_0",
                ),
            )

        self.assertEqual(captured["backend"], "gguf")
        self.assertEqual(captured["gguf_n_ctx"], 32768)
        self.assertEqual(captured["gguf_type_k"], "q8_0")
        self.assertEqual(captured["gguf_type_v"], "q4_0")
        self.assertEqual(
            entry["load_override"],
            {"gguf_n_ctx": 32768, "gguf_type_k": "Q8_0", "gguf_type_v": "q4_0"},
        )

    def test_load_model_applies_exllama_cache_overrides(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="exllamav3",
                models={
                    "exl-model": ModelSettings(
                        model_path="/models/exl3",
                        backend="exllamav3",
                        enabled=False,
                        exllama_cache_size=8192,
                        exllama_cache_quant="8,8",
                        exllama_max_rq_tokens=2048,
                    ),
                },
            ),
        )

        captured: dict[str, object] = {}

        class FakeBackend:
            def __init__(self):
                self._models = {"exl-model": object()}
                self._load_errors = {}

        engine = ModelRouterEngine(settings)

        def fake_build_backend(backend: str, scoped_settings: AppSettings):
            model_settings = scoped_settings.engine.models["exl-model"]
            captured["backend"] = backend
            captured["exllama_cache_size"] = model_settings.exllama_cache_size
            captured["exllama_cache_quant"] = model_settings.exllama_cache_quant
            captured["exllama_max_rq_tokens"] = model_settings.exllama_max_rq_tokens
            return FakeBackend()

        with mock.patch.object(engine, "_build_backend_engine", side_effect=fake_build_backend):
            entry = engine.load_model(
                "exl-model",
                settings,
                AdminLoadRequest(
                    exllama_cache_size=16384,
                    exllama_cache_quant=None,
                    exllama_max_rq_tokens=8192,
                ),
            )

        self.assertEqual(captured["backend"], "exllamav3")
        self.assertEqual(captured["exllama_cache_size"], 16384)
        self.assertIsNone(captured["exllama_cache_quant"])
        self.assertEqual(captured["exllama_max_rq_tokens"], 8192)
        self.assertEqual(
            entry["load_override"],
            {
                "exllama_cache_size": 16384,
                "exllama_cache_quant": None,
                "exllama_max_rq_tokens": 8192,
            },
        )

    def test_load_model_forces_enabled_on_scoped_settings(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "enabled-model": ModelSettings(model_path="/models/enabled"),
                    "disabled-model": ModelSettings(model_path="/models/disabled", enabled=False),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        captured: dict[str, bool] = {}

        class FakeBackend:
            def __init__(self):
                self._models = {"disabled-model": object()}
                self._load_errors = {}

        def fake_build_backend(backend: str, scoped_settings: AppSettings):
            del backend
            captured["enabled"] = scoped_settings.engine.models["disabled-model"].enabled
            return FakeBackend()

        with mock.patch.object(engine, "_build_backend_engine", side_effect=fake_build_backend):
            engine.load_model("disabled-model", settings)

        self.assertTrue(captured["enabled"])

    def test_load_model_is_idempotent_for_loaded_model(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "enabled-model": ModelSettings(model_path="/models/enabled"),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)
            with mock.patch.object(engine, "_build_backend_engine") as build_backend_engine:
                entry = engine.load_model("enabled-model", settings)

        self.assertEqual(entry["runtime_state"], "loaded")
        build_backend_engine.assert_not_called()

    def test_load_model_rejects_override_while_model_is_already_loaded(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="gguf",
                models={
                    "gguf-model": ModelSettings(model_path="/models/test.gguf", backend="gguf"),
                },
            ),
        )

        class FakeLlamaCppEngine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"gguf:{request.model}")

        with mock.patch.object(engine_module, "LlamaCppEngine", FakeLlamaCppEngine):
            engine = ModelRouterEngine(settings)

        with self.assertRaises(ValueError) as exc_info:
            engine.load_model("gguf-model", settings, AdminLoadRequest(gguf_n_ctx=32768))

        self.assertEqual(
            str(exc_info.exception),
            "load overrides can only be applied while the model is unloaded or failed; unload first",
        )

    def test_load_model_rejects_backend_mismatched_override(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="gguf",
                models={
                    "gguf-model": ModelSettings(model_path="/models/test.gguf", backend="gguf", enabled=False),
                },
            ),
        )

        engine = ModelRouterEngine(settings)

        with self.assertRaises(ValueError) as exc_info:
            engine.load_model("gguf-model", settings, AdminLoadRequest(exllama_cache_size=16384))

        self.assertEqual(
            str(exc_info.exception),
            "unsupported load override for gguf backend: exllama_cache_size",
        )

    def test_load_model_rejects_invalid_gguf_cache_type_override(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="gguf",
                models={
                    "gguf-model": ModelSettings(model_path="/models/test.gguf", backend="gguf", enabled=False),
                },
            ),
        )

        engine = ModelRouterEngine(settings)

        with self.assertRaises(ValueError) as exc_info:
            engine.load_model("gguf-model", settings, AdminLoadRequest(gguf_type_k="q8-0"))

        self.assertEqual(
            str(exc_info.exception),
            "GGUF cache type must contain only letters, digits, and underscores",
        )

    def test_load_model_rejects_unknown_gguf_cache_type_override(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="gguf",
                models={
                    "gguf-model": ModelSettings(model_path="/models/test.gguf", backend="gguf", enabled=False),
                },
            ),
        )

        fake_llama_cpp = types.ModuleType("llama_cpp")
        fake_llama_cpp.GGML_TYPE_Q8_0 = 17
        fake_llama_cpp.GGML_TYPE_Q4_0 = 18

        engine = ModelRouterEngine(settings)

        with mock.patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}):
            with self.assertRaises(ValueError) as exc_info:
                engine.load_model("gguf-model", settings, AdminLoadRequest(gguf_type_k="foo"))

        self.assertEqual(str(exc_info.exception), "unsupported GGUF cache type: 'foo'")

    def test_load_model_rejects_model_that_is_unloading(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "enabled-model": ModelSettings(model_path="/models/enabled"),
                    "other-model": ModelSettings(model_path="/models/other", enabled=False),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        engine._model_states["other-model"].lifecycle = "unloading"

        with self.assertRaises(engine_module.ModelStateError) as exc_info:
            engine.load_model("other-model", settings)

        self.assertEqual(exc_info.exception.code, "model_unloading")

    def test_load_model_marks_failure_and_retains_error(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "enabled-model": ModelSettings(model_path="/models/enabled"),
                    "broken-model": ModelSettings(model_path="/models/broken", enabled=False),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                if "broken-model" in scoped_settings.engine.models:
                    self._models = {}
                    self._load_errors = {"broken-model": "boom"}
                else:
                    self._models = {name: object() for name in scoped_settings.engine.models}
                    self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)
            with self.assertRaises(RuntimeError) as exc_info:
                engine.load_model("broken-model", settings)

        self.assertEqual(str(exc_info.exception), "boom")
        payload = engine.admin_models_payload()
        broken_model = payload["models"][1]
        self.assertEqual(broken_model["name"], "broken-model")
        self.assertEqual(broken_model["runtime_state"], "failed")
        self.assertEqual(broken_model["last_error"], "boom")

    def test_unload_model_removes_loaded_model_and_cleans_up(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "loaded-model": ModelSettings(model_path="/models/loaded"),
                    "other-model": ModelSettings(model_path="/models/other"),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)
            with mock.patch.object(engine, "_cleanup_runtime") as cleanup_runtime:
                entry = engine.unload_model("other-model", settings)

        self.assertEqual(entry["name"], "other-model")
        self.assertEqual(entry["runtime_state"], "unloaded")
        self.assertFalse(entry["is_loaded"])
        self.assertNotIn("other-model", engine._models)
        cleanup_runtime.assert_called_once()

    def test_unload_model_waits_for_inflight_and_blocks_new_requests(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "loaded-model": ModelSettings(model_path="/models/loaded"),
                    "other-model": ModelSettings(model_path="/models/other"),
                },
            ),
        )
        gate = threading.Event()
        entered = threading.Event()

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                if request.model == "other-model":
                    entered.set()
                    gate.wait(timeout=1.0)
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        result_holder: dict[str, EngineResult] = {}
        unload_holder: dict[str, dict[str, object]] = {}

        def run_complete() -> None:
            result_holder["result"] = engine.complete(ResponseRequest(model="other-model", input="hello"))

        def run_unload() -> None:
            unload_holder["entry"] = engine.unload_model("other-model", settings)

        request_thread = threading.Thread(target=run_complete)
        request_thread.start()
        entered.wait(timeout=1.0)

        unload_thread = threading.Thread(target=run_unload)
        unload_thread.start()

        for _ in range(50):
            payload = engine.admin_models_payload()
            other_model = payload["models"][1]
            if other_model["runtime_state"] == "unloading":
                break
            time.sleep(0.01)
        else:
            self.fail("model never entered unloading state")

        with self.assertRaises(engine_module.ModelStateError) as exc_info:
            engine.complete(ResponseRequest(model="other-model", input="hello again"))
        self.assertEqual(exc_info.exception.code, "model_unloading")

        gate.set()
        request_thread.join(timeout=1.0)
        unload_thread.join(timeout=1.0)

        self.assertEqual(result_holder["result"].text, "ct2:other-model")
        self.assertEqual(unload_holder["entry"]["runtime_state"], "unloaded")
        self.assertNotIn("other-model", engine._models)

    def test_admin_gpu_memory_payload_reports_gpu_and_model_estimates(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "enabled-model": ModelSettings(model_path="/models/enabled"),
                    "disabled-model": ModelSettings(model_path="/models/disabled", enabled=False),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with (
            mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine),
            mock.patch.object(
                router_module,
                "_query_gpu_memory",
                return_value=(
                    [
                        {
                            "index": 0,
                            "name": "GPU0",
                            "used_mib": 12000,
                            "total_mib": 24000,
                            "used_over_total": "12000MiB / 24000MiB",
                        }
                    ],
                    None,
                ),
            ),
            mock.patch.object(
                router_module,
                "_estimate_model_artifact_size_mib",
                side_effect=[4096, 2048],
            ),
        ):
            engine = ModelRouterEngine(settings)
            payload = engine.admin_gpu_memory_payload()

        self.assertEqual(payload["gpus"][0]["used_over_total"], "12000MiB / 24000MiB")
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["models"][0]["name"], "enabled-model")
        self.assertEqual(payload["models"][0]["vram_estimate_mib"], 4096)
        self.assertEqual(payload["models"][0]["vram_estimate_source"], "model_artifact_size")
        self.assertEqual(payload["models"][1]["name"], "disabled-model")
        self.assertEqual(payload["models"][1]["vram_estimate_mib"], 2048)
        self.assertEqual(payload["models"][1]["vram_estimate_source"], "model_artifact_size")


if __name__ == "__main__":
    unittest.main()
