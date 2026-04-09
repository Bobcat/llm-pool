from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_load_settings_reads_engine_model_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                (
                    "{\n"
                    '  "service": {"host": "0.0.0.0", "port": 9999, "log_level": "debug"},\n'
                    '  "engine": {\n'
                    '    "backend": "ct2",\n'
                    '    "default_model": "test-model",\n'
                    '    "models": {\n'
                    '      "test-model": {\n'
                    '        "model_path": "/models/test",\n'
                    '        "device": "cpu",\n'
                    '        "compute_type": "float32"\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(path)

        self.assertEqual(settings.service.host, "0.0.0.0")
        self.assertEqual(settings.service.port, 9999)
        self.assertEqual(settings.engine.backend, "ct2")
        self.assertEqual(settings.engine.default_model, "test-model")
        self.assertEqual(settings.engine.models["test-model"].model_path, "/models/test")
        self.assertEqual(settings.engine.models["test-model"].device, "cpu")
        self.assertEqual(settings.engine.models["test-model"].compute_type, "float32")

    def test_load_settings_applies_local_json_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            local_path = Path(tmpdir) / "local.json"
            settings_path.write_text(
                (
                    "{\n"
                    '  "service": {"host": "127.0.0.1", "port": 8011, "log_level": "info"},\n'
                    '  "engine": {\n'
                    '    "backend": "ct2",\n'
                    '    "default_model": "eurollm-9b-ct2-int8",\n'
                    '    "models": {\n'
                    '      "eurollm-9b-ct2-int8": {\n'
                    '        "model_path": "/models/eurollm",\n'
                    '        "device": "cuda",\n'
                    '        "compute_type": "int8"\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            local_path.write_text(
                (
                    "{\n"
                    '  "service": {"port": 18011},\n'
                    '  "engine": {\n'
                    '    "default_model": "new-model",\n'
                    '    "models": {\n'
                    '      "new-model": {\n'
                    '        "model_path": "/models/new",\n'
                    '        "device": "cuda",\n'
                    '        "compute_type": "float16"\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            previous_local_env = os.environ.get("LLM_POOL_LOCAL_SETTINGS_PATH")
            os.environ["LLM_POOL_LOCAL_SETTINGS_PATH"] = str(local_path)
            try:
                settings = load_settings(settings_path)
            finally:
                if previous_local_env is None:
                    os.environ.pop("LLM_POOL_LOCAL_SETTINGS_PATH", None)
                else:
                    os.environ["LLM_POOL_LOCAL_SETTINGS_PATH"] = previous_local_env

        self.assertEqual(settings.service.host, "127.0.0.1")
        self.assertEqual(settings.service.port, 18011)
        self.assertEqual(settings.engine.default_model, "new-model")
        self.assertIn("eurollm-9b-ct2-int8", settings.engine.models)
        self.assertIn("new-model", settings.engine.models)
