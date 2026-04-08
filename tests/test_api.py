from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None

if HAS_FASTAPI:
    from fastapi.testclient import TestClient


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ApiTests(unittest.TestCase):
    def _create_client(self) -> TestClient:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "engine": {"backend": "stub", "default_model": "test-model"}\n'
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
        self.assertIn("event: response.completed", response.text)

        events = [item for item in response.text.strip().split("\n\n") if item.strip()]
        completed = events[-1].split("data: ", 1)[1]
        completed_payload = json.loads(completed)
        self.assertEqual(completed_payload["output_text"], "Hello world")


if __name__ == "__main__":
    unittest.main()
