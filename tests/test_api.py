from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None

if HAS_FASTAPI:
    from fastapi.testclient import TestClient


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ApiTests(unittest.TestCase):
    def _create_client(self, settings_text: str | None = None) -> TestClient:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                settings_text
                or (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "test-model": {"model_path": "/tmp/test-model", "enabled": true},\n'
                    '      "disabled-model": {"model_path": "/tmp/disabled-model", "enabled": false}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            previous = os.environ.get("LLM_POOL_SETTINGS_PATH")
            os.environ["LLM_POOL_SETTINGS_PATH"] = str(settings_path)
            try:
                sys.modules.pop("app.main", None)
                main = importlib.import_module("app.main")
                app = main.create_app(settings_path)
            finally:
                if previous is None:
                    os.environ.pop("LLM_POOL_SETTINGS_PATH", None)
                else:
                    os.environ["LLM_POOL_SETTINGS_PATH"] = previous
        return TestClient(app)

    def test_json_response_mode_returns_response_envelope(self) -> None:
        client = self._create_client()

        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello world",
                "instructions": "Translate to Dutch.",
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["output"][0]["type"], "output_text")
        self.assertIn("Hello world", payload["output_text"])
        self.assertIn("metrics", payload)
        self.assertIn("engine_queue_wait_ms", payload["metrics"])
        self.assertIn("backend_inference_wall_ms", payload["metrics"])
        self.assertIn("engine_total_wall_ms", payload["metrics"])
        self.assertIn("engine_outside_backend_wall_ms", payload["metrics"])
        self.assertIn("pool_total_wall_ms", payload["metrics"])
        self.assertIn("gpu_generate_total_ms", payload["metrics"])

    def test_streaming_mode_returns_sse_events(self) -> None:
        client = self._create_client()

        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello world",
                "stream": True,
                "decoding": {
                    "beam_size": 1,
                    "top_k": 5,
                    "top_p": 0.9,
                    "temperature": 0.3,
                    "repetition_penalty": 1.0,
                    "max_tokens": 128,
                    "stop": ["<|im_end|>"],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: response.created", response.text)
        self.assertIn("event: response.output_text.delta", response.text)
        self.assertIn("event: response.metrics", response.text)
        self.assertIn("event: response.completed", response.text)

        events = [item for item in response.text.strip().split("\n\n") if item.strip()]
        metrics = events[-2].split("data: ", 1)[1]
        metrics_payload = json.loads(metrics)
        self.assertIn("metrics", metrics_payload)
        self.assertIn("engine_total_wall_ms", metrics_payload["metrics"])
        self.assertIn("pool_total_wall_ms", metrics_payload["metrics"])
        completed = events[-1].split("data: ", 1)[1]
        completed_payload = json.loads(completed)
        self.assertEqual(completed_payload["output_text"], "Hello world")

    def test_models_endpoint_returns_enabled_models(self) -> None:
        client = self._create_client()

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload, {"models": ["test-model"]})

    def test_admin_models_endpoint_returns_config_and_runtime_state(self) -> None:
        client = self._create_client()

        response = client.get("/v1/admin/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["models"]), 2)

        enabled_model = payload["models"][0]
        self.assertEqual(enabled_model["name"], "test-model")
        self.assertEqual(enabled_model["resolved_backend"], "stub")
        self.assertTrue(enabled_model["configured_enabled"])
        self.assertEqual(enabled_model["runtime_state"], "loaded")
        self.assertTrue(enabled_model["is_loaded"])
        self.assertEqual(enabled_model["replicas"], 1)
        self.assertEqual(enabled_model["replica_max"], 1)
        self.assertEqual(enabled_model["loaded_replicas"], 1)
        self.assertEqual(enabled_model["inflight_requests"], 0)
        self.assertEqual(enabled_model["queue_depth"], 0)
        self.assertEqual(enabled_model["runtime_inflight"], 0)
        self.assertEqual(enabled_model["configured_target_inflight"], 1)
        self.assertEqual(enabled_model["effective_target_inflight"], 1)
        self.assertIsNone(enabled_model["last_error"])
        self.assertIn("vram_estimate_mib", enabled_model)
        self.assertIn("vram_estimate_source", enabled_model)
        self.assertEqual(enabled_model["load_constraints"], {})
        self.assertEqual(enabled_model["load_recommendations"], {})
        self.assertEqual(enabled_model["definition"]["model_path"], "/tmp/test-model")
        self.assertTrue(enabled_model["definition"]["enabled"])
        self.assertNotIn("exllama_cache_size", enabled_model["definition"])
        self.assertNotIn("gguf_n_ctx", enabled_model["definition"])

        disabled_model = payload["models"][1]
        self.assertEqual(disabled_model["name"], "disabled-model")
        self.assertEqual(disabled_model["resolved_backend"], "stub")
        self.assertFalse(disabled_model["configured_enabled"])
        self.assertEqual(disabled_model["runtime_state"], "unloaded")
        self.assertFalse(disabled_model["is_loaded"])
        self.assertEqual(disabled_model["replicas"], 1)
        self.assertEqual(disabled_model["replica_max"], 1)
        self.assertEqual(disabled_model["loaded_replicas"], 0)
        self.assertEqual(disabled_model["inflight_requests"], 0)
        self.assertEqual(disabled_model["queue_depth"], 0)
        self.assertEqual(disabled_model["runtime_inflight"], 0)
        self.assertEqual(disabled_model["configured_target_inflight"], 1)
        self.assertEqual(disabled_model["effective_target_inflight"], 1)
        self.assertIsNone(disabled_model["last_error"])
        self.assertIn("vram_estimate_mib", disabled_model)
        self.assertIn("vram_estimate_source", disabled_model)
        self.assertEqual(disabled_model["load_constraints"], {})
        self.assertEqual(disabled_model["load_recommendations"], {})
        self.assertEqual(disabled_model["definition"]["model_path"], "/tmp/disabled-model")
        self.assertFalse(disabled_model["definition"]["enabled"])
        self.assertNotIn("exllama_cache_size", disabled_model["definition"])
        self.assertNotIn("gguf_n_ctx", disabled_model["definition"])

    def test_admin_gpu_memory_endpoint_returns_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "test-model": {"model_path": "/tmp/test-model", "enabled": true}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            previous = os.environ.get("LLM_POOL_SETTINGS_PATH")
            os.environ["LLM_POOL_SETTINGS_PATH"] = str(settings_path)
            try:
                sys.modules.pop("app.main", None)
                main = importlib.import_module("app.main")

                class FakeEngine:
                    def admin_models_payload(self, settings) -> dict[str, object]:
                        del settings
                        return {"models": []}

                    def admin_gpu_memory_payload(self, settings) -> dict[str, object]:
                        del settings
                        return {
                            "gpus": [
                                {
                                    "index": 0,
                                    "name": "Test GPU",
                                    "used_mib": 1234,
                                    "total_mib": 24000,
                                    "used_over_total": "1234MiB / 24000MiB",
                                }
                            ],
                            "models": [
                                {
                                    "name": "test-model",
                                    "runtime_state": "loaded",
                                    "is_loaded": True,
                                    "vram_estimate_mib": 2048,
                                    "vram_estimate_source": "model_artifact_size",
                                }
                            ],
                            "error": None,
                        }

                    def load_model(self, model_name: str, settings, load_request=None) -> dict[str, object]:
                        del model_name, settings, load_request
                        raise AssertionError("load_model should not be called in this test")

                    def unload_model(self, model_name: str, settings) -> dict[str, object]:
                        del model_name, settings
                        raise AssertionError("unload_model should not be called in this test")

                    def complete(self, request):
                        del request
                        raise AssertionError("complete should not be called in this test")

                with mock.patch.object(main, "build_engine", return_value=FakeEngine()):
                    app = main.create_app(settings_path)
            finally:
                if previous is None:
                    os.environ.pop("LLM_POOL_SETTINGS_PATH", None)
                else:
                    os.environ["LLM_POOL_SETTINGS_PATH"] = previous

        client = TestClient(app)
        response = client.get("/v1/admin/gpu-memory")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["gpus"][0]["used_over_total"], "1234MiB / 24000MiB")
        self.assertEqual(payload["models"][0]["name"], "test-model")
        self.assertEqual(payload["models"][0]["vram_estimate_source"], "model_artifact_size")
        self.assertIsNone(payload["error"])

    def test_admin_models_endpoint_reports_replica_configuration(self) -> None:
        client = self._create_client(
            (
                "{\n"
                '  "engine": {\n'
                '    "backend": "stub",\n'
                '    "models": {\n'
                '      "replica-model": {\n'
                '        "model_path": "/tmp/replica-model",\n'
                '        "enabled": true,\n'
                '        "replicas": 3,\n'
                '        "replica_max": 4\n'
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n"
            )
        )

        response = client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["replica-model"]})

        admin_response = client.get("/v1/admin/models")
        self.assertEqual(admin_response.status_code, 200)
        model = admin_response.json()["models"][0]
        self.assertEqual(model["name"], "replica-model")
        self.assertEqual(model["replicas"], 3)
        self.assertEqual(model["replica_max"], 4)
        self.assertEqual(model["loaded_replicas"], 3)

    def test_load_model_endpoint_loads_disabled_model_live(self) -> None:
        client = self._create_client()

        response = client.post("/v1/admin/models/disabled-model/load")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "disabled-model")
        self.assertEqual(payload["runtime_state"], "loaded")
        self.assertTrue(payload["is_loaded"])
        admin_response = client.get("/v1/admin/models")
        models = {item["name"]: item for item in admin_response.json()["models"]}
        self.assertEqual(models["disabled-model"]["runtime_state"], "loaded")

    def test_load_model_endpoint_forwards_optional_load_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "gguf",\n'
                    '    "models": {\n'
                    '      "gguf-model": {"model_path": "/tmp/test.gguf", "enabled": false, "backend": "gguf"}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            previous = os.environ.get("LLM_POOL_SETTINGS_PATH")
            os.environ["LLM_POOL_SETTINGS_PATH"] = str(settings_path)
            try:
                sys.modules.pop("app.main", None)
                main = importlib.import_module("app.main")
                captured: dict[str, object] = {}

                class FakeEngine:
                    def admin_models_payload(self, settings) -> dict[str, object]:
                        del settings
                        return {"models": []}

                    def admin_gpu_memory_payload(self, settings) -> dict[str, object]:
                        del settings
                        return {"gpus": [], "models": [], "error": None}

                    def load_model(self, model_name: str, settings, load_request=None) -> dict[str, object]:
                        del settings
                        captured["model_name"] = model_name
                        captured["load_request"] = load_request
                        return {
                            "name": model_name,
                            "resolved_backend": "gguf",
                            "configured_enabled": False,
                            "runtime_state": "loaded",
                            "is_loaded": True,
                            "replicas": 2,
                            "replica_max": 3,
                            "loaded_replicas": 2,
                            "inflight_requests": 0,
                            "queue_depth": 0,
                            "runtime_inflight": 0,
                            "configured_target_inflight": 1,
                            "effective_target_inflight": 1,
                            "last_error": None,
                            "vram_estimate_mib": None,
                            "vram_estimate_source": "unavailable",
                            "load_constraints": {
                                "gguf_n_ctx": {"kind": "integer", "minimum": 1, "step": 1},
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
                                    "allowed_values": [
                                        "f32",
                                        "f16",
                                        "bf16",
                                        "q8_0",
                                        "q4_0",
                                        "q4_1",
                                        "iq4_nl",
                                        "q5_0",
                                        "q5_1",
                                    ],
                                    "examples": ["f16", "q8_0", "q4_0"],
                                },
                                "gguf_type_v": {
                                    "kind": "string_or_null",
                                    "format": "ggml_type_name",
                                    "default": "f16",
                                    "allowed_values": [
                                        "f32",
                                        "f16",
                                        "bf16",
                                        "q8_0",
                                        "q4_0",
                                        "q4_1",
                                        "iq4_nl",
                                        "q5_0",
                                        "q5_1",
                                    ],
                                    "examples": ["f16", "q8_0", "q4_0"],
                                },
                            },
                            "load_recommendations": {
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
                            "load_override": {
                                "gguf_n_ctx": 32768,
                                "gguf_flash_attn": "auto",
                                "gguf_type_k": "q8_0",
                                "gguf_type_v": "q4_0",
                            },
                            "definition": {
                                "model_path": "/tmp/test.gguf",
                                "backend": "gguf",
                                "enabled": False,
                                "gguf_n_ctx": 4096,
                                "gguf_flash_attn": "auto",
                                "gguf_n_gpu_layers": -1,
                                "gguf_type_k": None,
                                "gguf_type_v": None,
                            },
                        }

                    def unload_model(self, model_name: str, settings) -> dict[str, object]:
                        del model_name, settings
                        raise AssertionError("unload_model should not be called in this test")

                    def complete(self, request):
                        del request
                        raise AssertionError("complete should not be called in this test")

                with mock.patch.object(main, "build_engine", return_value=FakeEngine()):
                    app = main.create_app(settings_path)
            finally:
                if previous is None:
                    os.environ.pop("LLM_POOL_SETTINGS_PATH", None)
                else:
                    os.environ["LLM_POOL_SETTINGS_PATH"] = previous

        client = TestClient(app)
        response = client.post(
            "/v1/admin/models/gguf-model/load",
            json={
                "replicas": 2,
                "gguf_n_ctx": 32768,
                "gguf_flash_attn": "auto",
                "gguf_type_k": "q8_0",
                "gguf_type_v": "q4_0",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["model_name"], "gguf-model")
        self.assertEqual(captured["load_request"].replicas, 2)
        self.assertEqual(captured["load_request"].gguf_n_ctx, 32768)
        self.assertEqual(captured["load_request"].gguf_flash_attn, "auto")
        self.assertEqual(captured["load_request"].gguf_type_k, "q8_0")
        self.assertEqual(captured["load_request"].gguf_type_v, "q4_0")
        self.assertEqual(response.json()["loaded_replicas"], 2)
        self.assertEqual(
            response.json()["load_override"],
            {"gguf_n_ctx": 32768, "gguf_flash_attn": "auto", "gguf_type_k": "q8_0", "gguf_type_v": "q4_0"},
        )

    def test_load_model_endpoint_rejects_unknown_model(self) -> None:
        client = self._create_client()

        response = client.post("/v1/admin/models/missing-model/load")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            {"code": "unknown_model", "model": "missing-model"},
        )

    def test_load_model_endpoint_reports_backend_load_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "enabled-model": {"model_path": "/tmp/enabled-model", "enabled": true},\n'
                    '      "broken-model": {"model_path": "/tmp/broken-model", "enabled": false}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            previous = os.environ.get("LLM_POOL_SETTINGS_PATH")
            os.environ["LLM_POOL_SETTINGS_PATH"] = str(settings_path)
            try:
                sys.modules.pop("app.main", None)
                main = importlib.import_module("app.main")

                class FakeFailingEngine:
                    def admin_models_payload(self, settings) -> dict[str, object]:
                        del settings
                        return {"models": []}

                    def load_model(self, model_name: str, settings, load_request=None) -> dict[str, object]:
                        del settings, load_request
                        raise RuntimeError(f"failed:{model_name}")

                    def complete(self, request):
                        del request
                        raise AssertionError("complete should not be called in this test")

                with mock.patch.object(main, "build_engine", return_value=FakeFailingEngine()):
                    app = main.create_app(settings_path)
            finally:
                if previous is None:
                    os.environ.pop("LLM_POOL_SETTINGS_PATH", None)
                else:
                    os.environ["LLM_POOL_SETTINGS_PATH"] = previous

        client = TestClient(app)
        response = client.post("/v1/admin/models/broken-model/load")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            {
                "code": "model_load_failed",
                "model": "broken-model",
                "message": "failed:broken-model",
            },
        )

    def test_load_model_endpoint_reports_invalid_load_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "gguf",\n'
                    '    "models": {\n'
                    '      "gguf-model": {"model_path": "/tmp/test.gguf", "enabled": false, "backend": "gguf"}\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            previous = os.environ.get("LLM_POOL_SETTINGS_PATH")
            os.environ["LLM_POOL_SETTINGS_PATH"] = str(settings_path)
            try:
                sys.modules.pop("app.main", None)
                main = importlib.import_module("app.main")

                class FakeEngine:
                    def admin_models_payload(self, settings) -> dict[str, object]:
                        del settings
                        return {"models": []}

                    def admin_gpu_memory_payload(self, settings) -> dict[str, object]:
                        del settings
                        return {"gpus": [], "models": [], "error": None}

                    def load_model(self, model_name: str, settings, load_request=None) -> dict[str, object]:
                        del model_name, settings, load_request
                        raise ValueError("unsupported load override for gguf backend: exllama_cache_size")

                    def unload_model(self, model_name: str, settings) -> dict[str, object]:
                        del model_name, settings
                        raise AssertionError("unload_model should not be called in this test")

                    def complete(self, request):
                        del request
                        raise AssertionError("complete should not be called in this test")

                with mock.patch.object(main, "build_engine", return_value=FakeEngine()):
                    app = main.create_app(settings_path)
            finally:
                if previous is None:
                    os.environ.pop("LLM_POOL_SETTINGS_PATH", None)
                else:
                    os.environ["LLM_POOL_SETTINGS_PATH"] = previous

        client = TestClient(app)
        response = client.post("/v1/admin/models/gguf-model/load", json={"exllama_cache_size": 16384})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            {
                "code": "invalid_load_request",
                "model": "gguf-model",
                "message": "unsupported load override for gguf backend: exllama_cache_size",
            },
        )

    def test_unload_model_endpoint_unloads_loaded_model(self) -> None:
        client = self._create_client(
            (
                "{\n"
                '  "engine": {\n'
                '    "backend": "stub",\n'
                '    "models": {\n'
                    '      "first-model": {"model_path": "/tmp/first-model", "enabled": true},\n'
                    '      "second-model": {"model_path": "/tmp/second-model", "enabled": true}\n'
                "    }\n"
                "  }\n"
                "}\n"
            )
        )

        response = client.post("/v1/admin/models/second-model/unload")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "second-model")
        self.assertEqual(payload["runtime_state"], "unloaded")
        self.assertFalse(payload["is_loaded"])
        inference_response = client.post(
            "/v1/responses",
            json={"model": "second-model", "input": "Hello", "stream": False},
        )
        self.assertEqual(inference_response.status_code, 409)
        self.assertEqual(
            inference_response.json()["detail"],
            {"code": "model_not_loaded", "model": "second-model"},
        )


if __name__ == "__main__":
    unittest.main()
