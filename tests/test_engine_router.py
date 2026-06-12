from __future__ import annotations

import builtins
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

        self.assertEqual(ct2_result.text, "ct2:ct2-model#1")
        self.assertEqual(exl_result.text, "exl3:exl-model#1")
        self.assertEqual(sorted(engine._models.keys()), ["ct2-model#1", "exl-model#1"])

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

        self.assertEqual(gguf_result.text, "gguf:gguf-model#1")

    def test_rejects_thinking_override_when_model_has_no_thinking_capability(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "gguf-model": ModelSettings(
                        model_path="/models/test.gguf",
                        backend="gguf",
                        prompt_format="generic",
                    ),
                },
            ),
        )

        class FakeLlamaCppEngine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"gguf:{request.model}")

        with mock.patch.object(engine_module, "LlamaCppEngine", FakeLlamaCppEngine):
            engine = ModelRouterEngine(settings)

            with self.assertRaises(engine_module.RequestAdmissionError) as exc_info:
                engine.complete(
                    ResponseRequest(
                        model="gguf-model",
                        input="hello",
                        thinking="enabled",
                    )
                )

        self.assertEqual(exc_info.exception.code, "thinking_unsupported")

    def test_dispatches_openai_compatible_backend_with_remote_admission(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "remote-model": ModelSettings(
                        model_path=None,
                        backend="openai_compatible",
                        remote_api_kind="chat_completions",
                        remote_base_url="https://api.example.com/v1",
                        remote_api_key_env="EXAMPLE_API_KEY",
                        remote_model="provider-model",
                        target_inflight=3,
                    ),
                },
            ),
        )

        class FakeOpenAICompatibleEngine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"remote:{request.model}")

        with mock.patch.object(engine_module, "OpenAICompatibleEngine", FakeOpenAICompatibleEngine):
            engine = ModelRouterEngine(settings)

            with self.assertRaises(engine_module.RequestAdmissionError) as exc_info:
                engine.complete(ResponseRequest(model="remote-model", input="hello"))

            remote_result = engine.complete(
                ResponseRequest(
                    model="remote-model",
                    input="hello",
                    allow_remote=True,
                )
            )

        self.assertEqual(exc_info.exception.code, "remote_execution_disallowed")
        self.assertEqual(remote_result.text, "remote:remote-model#1")
        model = engine.admin_models_payload()["models"][0]
        self.assertEqual(model["effective_target_inflight"], 3)

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
                self._models = {}
                self._load_errors = {}
                for model_name in scoped_settings.engine.models:
                    if model_name.startswith("broken-model#"):
                        self._load_errors[model_name] = "boom"
                    else:
                        self._models[model_name] = object()

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
        self.assertEqual(loaded_model["replicas"], 1)
        self.assertEqual(loaded_model["replica_max"], 1)
        self.assertEqual(loaded_model["loaded_replicas"], 1)
        self.assertIsNone(loaded_model["last_error"])
        self.assertEqual(loaded_model["queue_depth"], 0)
        self.assertEqual(loaded_model["runtime_inflight"], 0)
        self.assertEqual(loaded_model["configured_target_inflight"], 1)
        self.assertEqual(loaded_model["effective_target_inflight"], 1)
        self.assertIn("vram_estimate_mib", loaded_model)
        self.assertIn("vram_estimate_source", loaded_model)
        self.assertEqual(loaded_model["load_constraints"], {})
        self.assertEqual(loaded_model["load_recommendations"], {})
        self.assertIn("device", loaded_model["definition"])
        self.assertIn("compute_type", loaded_model["definition"])
        self.assertNotIn("exllama_cache_size", loaded_model["definition"])
        self.assertNotIn("gguf_n_ctx", loaded_model["definition"])

        failed_model = payload["models"][1]
        self.assertEqual(failed_model["name"], "broken-model")
        self.assertEqual(failed_model["runtime_state"], "failed")
        self.assertFalse(failed_model["is_loaded"])
        self.assertEqual(failed_model["replicas"], 1)
        self.assertEqual(failed_model["replica_max"], 1)
        self.assertEqual(failed_model["loaded_replicas"], 0)
        self.assertEqual(failed_model["last_error"], "broken-model#1: boom")
        self.assertEqual(failed_model["queue_depth"], 0)
        self.assertEqual(failed_model["runtime_inflight"], 0)
        self.assertEqual(failed_model["configured_target_inflight"], 1)
        self.assertEqual(failed_model["effective_target_inflight"], 1)
        self.assertIn("vram_estimate_mib", failed_model)
        self.assertIn("vram_estimate_source", failed_model)
        self.assertEqual(failed_model["load_constraints"], {})
        self.assertEqual(failed_model["load_recommendations"], {})
        self.assertIn("device", failed_model["definition"])
        self.assertIn("compute_type", failed_model["definition"])
        self.assertNotIn("exllama_cache_size", failed_model["definition"])
        self.assertNotIn("gguf_n_ctx", failed_model["definition"])

        unloaded_model = payload["models"][2]
        self.assertEqual(unloaded_model["name"], "disabled-model")
        self.assertEqual(unloaded_model["runtime_state"], "unloaded")
        self.assertFalse(unloaded_model["is_loaded"])
        self.assertEqual(unloaded_model["replicas"], 1)
        self.assertEqual(unloaded_model["replica_max"], 1)
        self.assertEqual(unloaded_model["loaded_replicas"], 0)
        self.assertIsNone(unloaded_model["last_error"])
        self.assertEqual(unloaded_model["queue_depth"], 0)
        self.assertEqual(unloaded_model["runtime_inflight"], 0)
        self.assertEqual(unloaded_model["configured_target_inflight"], 1)
        self.assertEqual(unloaded_model["effective_target_inflight"], 1)
        self.assertIn("vram_estimate_mib", unloaded_model)
        self.assertIn("vram_estimate_source", unloaded_model)
        self.assertEqual(unloaded_model["load_constraints"], {})
        self.assertEqual(unloaded_model["load_recommendations"], {})
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
                "gguf_flash_attn": {
                    "kind": "enum",
                    "default": "auto",
                    "allowed_values": ["on", "off", "auto"],
                    "examples": ["auto", "on", "off"],
                },
                "gguf_type_k": {
                    "kind": "string_or_null",
                    "format": "ggml_type_name",
                    "default": "f16",
                    "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                    "examples": ["f16", "q8_0", "q4_0"],
                },
                "gguf_type_v": {
                    "kind": "string_or_null",
                    "format": "ggml_type_name",
                    "default": "f16",
                    "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                    "examples": ["f16", "q8_0", "q4_0"],
                },
            },
        )
        self.assertIn("gguf_n_ctx", payload["models"][0]["definition"])
        self.assertIn("gguf_flash_attn", payload["models"][0]["definition"])
        self.assertIn("gguf_type_k", payload["models"][0]["definition"])
        self.assertIn("gguf_type_v", payload["models"][0]["definition"])
        self.assertEqual(
            payload["models"][0]["load_recommendations"],
            {
                "gguf_cache_type_pairs": {
                    "kind": "pair_presets",
                    "fields": ["gguf_type_k", "gguf_type_v"],
                    "recommended_pairs": [
                        {
                            "label": "f16/f16",
                            "gguf_type_k": "f16",
                            "gguf_type_v": "f16",
                        },
                        {
                            "label": "q8_0/q8_0",
                            "gguf_type_k": "q8_0",
                            "gguf_type_v": "q8_0",
                        },
                        {
                            "label": "q4_0/q4_0",
                            "gguf_type_k": "q4_0",
                            "gguf_type_v": "q4_0",
                        },
                    ],
                    "notes": [
                        "Service-curated presets for GGUF cache types.",
                        "Prefer symmetric GGUF K/V pairs by default; asymmetric pairs may reduce or disable GPU offload in upstream llama.cpp.",
                    ],
                }
            },
        )
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
                "exllama_cache_k_bits": {
                    "kind": "integer_or_null",
                    "minimum": 2,
                    "maximum": 8,
                    "default": None,
                    "null_means": "fp16",
                    "allowed_values": [2, 3, 4, 5, 6, 7, 8],
                },
                "exllama_cache_v_bits": {
                    "kind": "integer_or_null",
                    "minimum": 2,
                    "maximum": 8,
                    "default": None,
                    "null_means": "fp16",
                    "allowed_values": [2, 3, 4, 5, 6, 7, 8],
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
        self.assertEqual(
            payload["models"][0]["load_recommendations"],
            {
                "exllama_cache_bit_pairs": {
                    "kind": "pair_presets",
                    "fields": ["exllama_cache_k_bits", "exllama_cache_v_bits"],
                    "recommended_pairs": [
                        {
                            "label": "fp16",
                            "exllama_cache_k_bits": None,
                            "exllama_cache_v_bits": None,
                        },
                        {
                            "label": "8/8",
                            "exllama_cache_k_bits": 8,
                            "exllama_cache_v_bits": 8,
                        },
                        {
                            "label": "8/4",
                            "exllama_cache_k_bits": 8,
                            "exllama_cache_v_bits": 4,
                        },
                    ],
                    "notes": [
                        "Service-curated presets for ExLlamaV3 cache bits.",
                        "These presets are not an exhaustive list of valid ExLlamaV3 K/V bit pairs.",
                    ],
                }
            },
        )
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
                self._models = {name: object() for name in scoped_settings.engine.models}
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
        self.assertEqual(payload["models"][0]["queue_depth"], 0)
        self.assertEqual(payload["models"][0]["runtime_inflight"], 1)

        gate.set()
        thread.join(timeout=1.0)

        self.assertEqual(result_holder["result"].text, "ct2:ct2-model#1")
        self.assertIsNotNone(result_holder["result"].metrics.engine_queue_wait_ms)
        self.assertIsNotNone(result_holder["result"].metrics.backend_inference_wall_ms)
        self.assertIsNotNone(result_holder["result"].metrics.engine_total_wall_ms)
        self.assertIsNotNone(result_holder["result"].metrics.engine_outside_backend_wall_ms)
        self.assertIsNotNone(result_holder["result"].metrics.pool_total_wall_ms)
        self.assertGreaterEqual(result_holder["result"].metrics.engine_total_wall_ms, 0.0)
        self.assertGreaterEqual(result_holder["result"].metrics.pool_total_wall_ms, 0.0)
        self.assertGreaterEqual(result_holder["result"].metrics.backend_inference_wall_ms, 0.0)
        payload = engine.admin_models_payload()
        self.assertEqual(payload["models"][0]["inflight_requests"], 0)
        self.assertEqual(payload["models"][0]["queue_depth"], 0)
        self.assertEqual(payload["models"][0]["runtime_inflight"], 0)

    def test_scheduler_tracks_queued_work_separately_from_runtime_inflight(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "ct2-model": ModelSettings(model_path="/models/ct2", target_inflight=4),
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
                entered.set()
                gate.wait(timeout=1.0)
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        results: list[EngineResult] = []

        def run_complete() -> None:
            result = engine.complete(ResponseRequest(model="ct2-model", input="hello"))
            results.append(result)

        first_thread = threading.Thread(target=run_complete)
        second_thread = threading.Thread(target=run_complete)
        first_thread.start()
        entered.wait(timeout=1.0)
        second_thread.start()
        time.sleep(0.05)

        payload = engine.admin_models_payload()["models"][0]
        self.assertEqual(payload["inflight_requests"], 2)
        self.assertEqual(payload["queue_depth"], 1)
        self.assertEqual(payload["runtime_inflight"], 1)
        self.assertEqual(payload["configured_target_inflight"], 4)
        self.assertEqual(payload["effective_target_inflight"], 1)

        gate.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)
        self.assertEqual([result.text for result in results], ["ct2:ct2-model#1", "ct2:ct2-model#1"])
        self.assertTrue(all(result.metrics.engine_total_wall_ms is not None for result in results))
        self.assertTrue(all(result.metrics.pool_total_wall_ms is not None for result in results))
        self.assertTrue(all(result.metrics.backend_inference_wall_ms is not None for result in results))
        self.assertTrue(any((result.metrics.engine_queue_wait_ms or 0.0) > 0.0 for result in results))
        self.assertTrue(
            all(
                (result.metrics.engine_total_wall_ms or 0.0) >= (result.metrics.backend_inference_wall_ms or 0.0)
                for result in results
            )
        )
        self.assertTrue(
            all(
                (result.metrics.pool_total_wall_ms or 0.0) >= (result.metrics.engine_total_wall_ms or 0.0)
                for result in results
            )
        )

    def test_unload_cancels_queued_work_and_drains_running_work(self) -> None:
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
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                entered.set()
                gate.wait(timeout=1.0)
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        running_result: dict[str, str] = {}
        queued_error: dict[str, str] = {}
        unload_result: dict[str, dict[str, object]] = {}

        def run_first() -> None:
            running_result["text"] = engine.complete(ResponseRequest(model="ct2-model", input="first")).text

        def run_second() -> None:
            try:
                engine.complete(ResponseRequest(model="ct2-model", input="second"))
            except engine_module.ModelStateError as exc:
                queued_error["code"] = exc.code

        def run_unload() -> None:
            unload_result["entry"] = engine.unload_model("ct2-model", settings)

        first_thread = threading.Thread(target=run_first)
        second_thread = threading.Thread(target=run_second)
        unload_thread = threading.Thread(target=run_unload)
        first_thread.start()
        entered.wait(timeout=1.0)
        second_thread.start()
        time.sleep(0.05)
        unload_thread.start()
        time.sleep(0.05)

        payload = engine.admin_models_payload()["models"][0]
        self.assertEqual(payload["runtime_state"], "unloading")
        self.assertEqual(payload["queue_depth"], 0)
        self.assertEqual(payload["runtime_inflight"], 1)

        gate.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)
        unload_thread.join(timeout=1.0)

        self.assertEqual(running_result["text"], "ct2:ct2-model#1")
        self.assertEqual(queued_error["code"], "model_unloading")
        self.assertEqual(unload_result["entry"]["runtime_state"], "unloaded")

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
        self.assertEqual(entry["loaded_replicas"], 1)
        self.assertIn("disabled-model#1", engine._models)
        self.assertEqual(engine.admin_models_payload()["models"][1]["runtime_state"], "loaded")

    def test_load_model_can_load_configured_replica_group(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "replica-model": ModelSettings(
                        model_path="/models/replica",
                        enabled=False,
                        replicas=2,
                        replica_max=3,
                    ),
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
            entry = engine.load_model("replica-model", settings)

        self.assertEqual(entry["runtime_state"], "loaded")
        self.assertEqual(entry["replicas"], 2)
        self.assertEqual(entry["replica_max"], 3)
        self.assertEqual(entry["loaded_replicas"], 2)
        self.assertEqual(sorted(engine._models.keys()), ["replica-model#1", "replica-model#2"])
        self.assertEqual(engine.list_models_payload(), {"models": ["replica-model"]})

    def test_load_model_can_override_replica_count_while_unloaded(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "replica-model": ModelSettings(
                        model_path="/models/replica",
                        enabled=False,
                        replicas=1,
                        replica_max=3,
                    ),
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
            entry = engine.load_model("replica-model", settings, AdminLoadRequest(replicas=2))

        self.assertEqual(entry["replicas"], 2)
        self.assertEqual(entry["replica_max"], 3)
        self.assertEqual(entry["loaded_replicas"], 2)
        self.assertEqual(sorted(engine._models.keys()), ["replica-model#1", "replica-model#2"])

    def test_scheduler_distributes_work_across_loaded_replicas(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                models={
                    "replica-model": ModelSettings(
                        model_path="/models/replica",
                        replicas=2,
                        replica_max=2,
                    ),
                },
            ),
        )
        gate = threading.Event()
        entered_replicas: list[str] = []
        entered_lock = threading.Lock()

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}
                self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                with entered_lock:
                    entered_replicas.append(request.model)
                gate.wait(timeout=1.0)
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)

        results: list[EngineResult] = []

        def run_complete() -> None:
            results.append(engine.complete(ResponseRequest(model="replica-model", input="hello")))

        first_thread = threading.Thread(target=run_complete)
        second_thread = threading.Thread(target=run_complete)
        first_thread.start()
        second_thread.start()

        for _ in range(50):
            payload = engine.admin_models_payload()["models"][0]
            if payload["runtime_inflight"] == 2:
                break
            time.sleep(0.01)
        else:
            self.fail("replica group never reached runtime_inflight=2")

        payload = engine.admin_models_payload()["models"][0]
        self.assertEqual(payload["loaded_replicas"], 2)
        self.assertEqual(payload["queue_depth"], 0)
        self.assertEqual(payload["runtime_inflight"], 2)

        gate.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)

        self.assertEqual(sorted(entered_replicas), ["replica-model#1", "replica-model#2"])
        self.assertEqual(sorted(result.text for result in results), ["ct2:replica-model#1", "ct2:replica-model#2"])

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
                        gguf_flash_attn="off",
                        gguf_type_k="f16",
                        gguf_type_v="f16",
                    ),
                },
            ),
        )

        captured: dict[str, object] = {}

        class FakeBackend:
            def __init__(self):
                self._models = {}
                self._load_errors = {}

        engine = ModelRouterEngine(settings)

        def fake_build_backend(backend: str, scoped_settings: AppSettings):
            model_settings = next(iter(scoped_settings.engine.models.values()))
            captured["backend"] = backend
            captured["replica_ids"] = sorted(scoped_settings.engine.models.keys())
            captured["gguf_n_ctx"] = model_settings.gguf_n_ctx
            captured["gguf_flash_attn"] = model_settings.gguf_flash_attn
            captured["gguf_type_k"] = model_settings.gguf_type_k
            captured["gguf_type_v"] = model_settings.gguf_type_v
            backend_instance = FakeBackend()
            backend_instance._models = {name: object() for name in scoped_settings.engine.models}
            return backend_instance

        with mock.patch.object(engine, "_build_backend_engine", side_effect=fake_build_backend):
            entry = engine.load_model(
                "gguf-model",
                settings,
                AdminLoadRequest(
                    gguf_n_ctx=32768,
                    gguf_flash_attn="AUTO",
                    gguf_type_k="Q8_0",
                    gguf_type_v="q4_0",
                ),
            )

        self.assertEqual(captured["backend"], "gguf")
        self.assertEqual(captured["replica_ids"], ["gguf-model#1"])
        self.assertEqual(captured["gguf_n_ctx"], 32768)
        self.assertEqual(captured["gguf_flash_attn"], "auto")
        self.assertEqual(captured["gguf_type_k"], "q8_0")
        self.assertEqual(captured["gguf_type_v"], "q4_0")
        self.assertEqual(
            entry["load_override"],
            {"gguf_n_ctx": 32768, "gguf_flash_attn": "auto", "gguf_type_k": "Q8_0", "gguf_type_v": "q4_0"},
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
                self._models = {}
                self._load_errors = {}

        engine = ModelRouterEngine(settings)

        def fake_build_backend(backend: str, scoped_settings: AppSettings):
            model_settings = next(iter(scoped_settings.engine.models.values()))
            captured["backend"] = backend
            captured["replica_ids"] = sorted(scoped_settings.engine.models.keys())
            captured["exllama_cache_size"] = model_settings.exllama_cache_size
            captured["exllama_cache_quant"] = model_settings.exllama_cache_quant
            captured["exllama_max_rq_tokens"] = model_settings.exllama_max_rq_tokens
            backend_instance = FakeBackend()
            backend_instance._models = {name: object() for name in scoped_settings.engine.models}
            return backend_instance

        with mock.patch.object(engine, "_build_backend_engine", side_effect=fake_build_backend):
            entry = engine.load_model(
                "exl-model",
                settings,
                AdminLoadRequest(
                    exllama_cache_size=16384,
                    exllama_cache_k_bits=8,
                    exllama_cache_v_bits=4,
                    exllama_max_rq_tokens=8192,
                ),
            )

        self.assertEqual(captured["backend"], "exllamav3")
        self.assertEqual(captured["replica_ids"], ["exl-model#1"])
        self.assertEqual(captured["exllama_cache_size"], 16384)
        self.assertEqual(captured["exllama_cache_quant"], "8,4")
        self.assertEqual(captured["exllama_max_rq_tokens"], 8192)
        self.assertEqual(
            entry["load_override"],
            {
                "exllama_cache_size": 16384,
                "exllama_cache_quant": "8,4",
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

        captured: dict[str, object] = {}

        class FakeBackend:
            def __init__(self):
                self._models = {}
                self._load_errors = {}

        def fake_build_backend(backend: str, scoped_settings: AppSettings):
            del backend
            model_settings = next(iter(scoped_settings.engine.models.values()))
            captured["replica_ids"] = sorted(scoped_settings.engine.models.keys())
            captured["enabled"] = model_settings.enabled
            backend_instance = FakeBackend()
            backend_instance._models = {name: object() for name in scoped_settings.engine.models}
            return backend_instance

        with mock.patch.object(engine, "_build_backend_engine", side_effect=fake_build_backend):
            engine.load_model("disabled-model", settings)

        self.assertEqual(captured["replica_ids"], ["disabled-model#1"])
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
            "replica count and load overrides can only be applied while the model is unloaded or failed; unload first",
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

    def test_load_model_rejects_invalid_gguf_flash_attn_override(self) -> None:
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
            engine.load_model("gguf-model", settings, AdminLoadRequest(gguf_flash_attn="sometimes"))

        self.assertEqual(
            str(exc_info.exception),
            "gguf_flash_attn must be one of: on, off, auto",
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

    def test_load_model_rejects_partial_exllama_cache_bits_override(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="exllamav3",
                models={
                    "exl-model": ModelSettings(model_path="/models/exl3", backend="exllamav3", enabled=False),
                },
            ),
        )

        engine = ModelRouterEngine(settings)

        with self.assertRaises(ValueError) as exc_info:
            engine.load_model("exl-model", settings, AdminLoadRequest(exllama_cache_k_bits=8))

        self.assertEqual(
            str(exc_info.exception),
            "exllama_cache_k_bits and exllama_cache_v_bits must be provided together",
        )

    def test_load_model_rejects_mixed_exllama_quant_and_cache_bits_override(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="exllamav3",
                models={
                    "exl-model": ModelSettings(model_path="/models/exl3", backend="exllamav3", enabled=False),
                },
            ),
        )

        engine = ModelRouterEngine(settings)

        with self.assertRaises(ValueError) as exc_info:
            engine.load_model(
                "exl-model",
                settings,
                AdminLoadRequest(exllama_cache_quant="8,8", exllama_cache_k_bits=8, exllama_cache_v_bits=4),
            )

        self.assertEqual(
            str(exc_info.exception),
            "exllama_cache_quant cannot be combined with exllama_cache_k_bits/exllama_cache_v_bits",
        )

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
                if any(model_name.startswith("broken-model#") for model_name in scoped_settings.engine.models):
                    self._models = {}
                    self._load_errors = {name: "boom" for name in scoped_settings.engine.models}
                else:
                    self._models = {name: object() for name in scoped_settings.engine.models}
                    self._load_errors = {}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        with mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine):
            engine = ModelRouterEngine(settings)
            with self.assertRaises(RuntimeError) as exc_info:
                engine.load_model("broken-model", settings)

        self.assertEqual(str(exc_info.exception), "broken-model#1: boom")
        payload = engine.admin_models_payload()
        broken_model = payload["models"][1]
        self.assertEqual(broken_model["name"], "broken-model")
        self.assertEqual(broken_model["runtime_state"], "failed")
        self.assertEqual(broken_model["last_error"], "broken-model#1: boom")

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
        self.assertNotIn("other-model#1", engine._models)
        cleanup_runtime.assert_called_once()

    def test_cleanup_gguf_runtime_does_not_import_exllamav3(self) -> None:
        class FakeLlama:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class FakeGgufRuntime:
            def __init__(self, llm: FakeLlama) -> None:
                self.llm = llm

        fake_llm = FakeLlama()
        runtime = FakeGgufRuntime(fake_llm)
        engine = ModelRouterEngine.__new__(ModelRouterEngine)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("exllamav3"):
                raise AssertionError(
                    f"unexpected ExLlamaV3 import during GGUF cleanup: {name}"
                )
            return original_import(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=guarded_import),
            mock.patch.object(router_module, "_empty_cuda_allocator_cache"),
            mock.patch.object(router_module.gc, "collect"),
        ):
            engine._cleanup_runtime(runtime)

        self.assertTrue(fake_llm.closed)
        self.assertIsNone(runtime.llm)

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
                if request.model == "other-model#1":
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

        self.assertEqual(result_holder["result"].text, "ct2:other-model#1")
        self.assertEqual(unload_holder["entry"]["runtime_state"], "unloaded")
        self.assertNotIn("other-model#1", engine._models)

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
