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
        self.assertEqual(enabled_model["inflight_requests"], 0)
        self.assertIsNone(enabled_model["last_error"])
        self.assertIn("vram_estimate_mib", enabled_model)
        self.assertIn("vram_estimate_source", enabled_model)
        self.assertEqual(enabled_model["definition"]["model_path"], "/tmp/test-model")
        self.assertTrue(enabled_model["definition"]["enabled"])

        disabled_model = payload["models"][1]
        self.assertEqual(disabled_model["name"], "disabled-model")
        self.assertEqual(disabled_model["resolved_backend"], "stub")
        self.assertFalse(disabled_model["configured_enabled"])
        self.assertEqual(disabled_model["runtime_state"], "unloaded")
        self.assertFalse(disabled_model["is_loaded"])
        self.assertEqual(disabled_model["inflight_requests"], 0)
        self.assertIsNone(disabled_model["last_error"])
        self.assertIn("vram_estimate_mib", disabled_model)
        self.assertIn("vram_estimate_source", disabled_model)
        self.assertEqual(disabled_model["definition"]["model_path"], "/tmp/disabled-model")
        self.assertFalse(disabled_model["definition"]["enabled"])

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

                    def load_model(self, model_name: str, settings) -> dict[str, object]:
                        del model_name, settings
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

                    def load_model(self, model_name: str, settings) -> dict[str, object]:
                        del settings
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
