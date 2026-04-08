from __future__ import annotations

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
