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
                    '    "decoding": {\n'
                    '      "beam_size": 2,\n'
                    '      "top_k": 3,\n'
                    '      "top_p": 0.7,\n'
                    '      "temperature": 0.2,\n'
                    '      "repetition_penalty": 1.1,\n'
                    '      "max_tokens": 300,\n'
                    '      "stop": ["</stop>"]\n'
                    "    },\n"
                    '    "models": {\n'
                    '      "test-model": {\n'
                    '        "model_path": "/models/test",\n'
                    '        "device": "cpu",\n'
                    '        "compute_type": "float32",\n'
                    '        "prompt_format": "qwen3_template",\n'
                    '        "enable_thinking": false,\n'
                    '        "target_inflight": 3,\n'
                    '        "enabled": false\n'
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
        self.assertEqual(settings.engine.models["test-model"].model_path, "/models/test")
        self.assertEqual(settings.engine.models["test-model"].device, "cpu")
        self.assertEqual(settings.engine.models["test-model"].compute_type, "float32")
        self.assertEqual(settings.engine.models["test-model"].prompt_format, "qwen3_template")
        self.assertFalse(settings.engine.models["test-model"].enable_thinking)
        self.assertEqual(settings.engine.models["test-model"].replicas, 1)
        self.assertEqual(settings.engine.models["test-model"].replica_max, 1)
        self.assertEqual(settings.engine.models["test-model"].target_inflight, 3)
        self.assertFalse(settings.engine.models["test-model"].enabled)
        self.assertEqual(settings.engine.decoding.beam_size, 2)
        self.assertEqual(settings.engine.decoding.top_k, 3)
        self.assertEqual(settings.engine.decoding.top_p, 0.7)
        self.assertEqual(settings.engine.decoding.temperature, 0.2)
        self.assertEqual(settings.engine.decoding.repetition_penalty, 1.1)
        self.assertEqual(settings.engine.decoding.max_tokens, 300)
        self.assertEqual(settings.engine.decoding.stop, ["</stop>"])

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
                    '    "decoding": {\n'
                    '      "beam_size": 1,\n'
                    '      "top_k": 1,\n'
                    '      "top_p": 1.0,\n'
                    '      "temperature": 0.1,\n'
                    '      "repetition_penalty": 1.0,\n'
                    '      "max_tokens": 256,\n'
                    '      "stop": ["<|im_end|>"]\n'
                    "    },\n"
                    '    "models": {\n'
                    '      "eurollm-9b-ct2-int8": {\n'
                    '        "model_path": "/models/eurollm",\n'
                    '        "device": "cuda",\n'
                    '        "compute_type": "int8",\n'
                    '        "prompt_format": "generic",\n'
                    '        "enabled": true\n'
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
                    '    "decoding": {\n'
                    '      "top_k": 7,\n'
                    '      "temperature": 0.4,\n'
                    '      "stop": ["</custom>"]\n'
                    "    },\n"
                    '    "models": {\n'
                    '      "new-model": {\n'
                    '        "model_path": "/models/new",\n'
                    '        "device": "cuda",\n'
                    '        "compute_type": "float16",\n'
                    '        "prompt_format": "qwen3_template",\n'
                    '        "enable_thinking": false,\n'
                    '        "enabled": false\n'
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
        self.assertIn("eurollm-9b-ct2-int8", settings.engine.models)
        self.assertIn("new-model", settings.engine.models)
        self.assertEqual(settings.engine.models["new-model"].prompt_format, "qwen3_template")
        self.assertFalse(settings.engine.models["new-model"].enable_thinking)
        self.assertTrue(settings.engine.models["eurollm-9b-ct2-int8"].enabled)
        self.assertFalse(settings.engine.models["new-model"].enabled)
        self.assertEqual(settings.engine.decoding.beam_size, 1)
        self.assertEqual(settings.engine.decoding.top_k, 7)
        self.assertEqual(settings.engine.decoding.top_p, 1.0)
        self.assertEqual(settings.engine.decoding.temperature, 0.4)
        self.assertEqual(settings.engine.decoding.stop, ["</custom>"])

    def test_load_settings_defaults_stop_list_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "ct2",\n'
                    '    "models": {\n'
                    '      "test-model": {\n'
                    '        "model_path": "/models/test"\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(path)

        self.assertEqual(settings.engine.decoding.stop, [])

    def test_load_settings_reads_model_backend_specific_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "ct2",\n'
                    '    "models": {\n'
                    '      "mixtral-exl3": {\n'
                    '        "model_path": "/models/mixtral-exl3",\n'
                    '        "backend": "exllamav3",\n'
                    '        "exllama_cache_size": 16384,\n'
                    '        "exllama_cache_quant": "8,4",\n'
                    '        "exllama_gpu_split": "24,24",\n'
                    '        "exllama_tensor_parallel": true,\n'
                    '        "exllama_tp_backend": "native",\n'
                    '        "exllama_max_batch_size": 32,\n'
                    '        "exllama_max_chunk_size": 1024,\n'
                    '        "exllama_max_q_size": 6,\n'
                    '        "exllama_max_rq_tokens": 2048\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(path)

        model = settings.engine.models["mixtral-exl3"]
        self.assertEqual(model.backend, "exllamav3")
        self.assertEqual(model.exllama_cache_size, 16384)
        self.assertEqual(model.exllama_cache_quant, "8,4")
        self.assertEqual(model.exllama_gpu_split, "24,24")
        self.assertTrue(model.exllama_tensor_parallel)
        self.assertEqual(model.exllama_tp_backend, "native")
        self.assertEqual(model.exllama_max_batch_size, 32)
        self.assertEqual(model.exllama_max_chunk_size, 1024)
        self.assertEqual(model.exllama_max_q_size, 6)
        self.assertEqual(model.exllama_max_rq_tokens, 2048)

    def test_load_settings_reads_gguf_backend_specific_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "gguf",\n'
                    '    "models": {\n'
                    '      "test-gguf": {\n'
                    '        "model_path": "/models/test.gguf",\n'
                    '        "backend": "gguf",\n'
                    '        "gguf_n_gpu_layers": 42,\n'
                    '        "gguf_n_ctx": 8192,\n'
                    '        "gguf_flash_attn": "off",\n'
                    '        "gguf_type_k": "q8_0",\n'
                    '        "gguf_type_v": "q4_0"\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(path)

        model = settings.engine.models["test-gguf"]
        self.assertEqual(model.backend, "gguf")
        self.assertEqual(model.gguf_n_gpu_layers, 42)
        self.assertEqual(model.gguf_n_ctx, 8192)
        self.assertEqual(model.gguf_flash_attn, "off")
        self.assertEqual(model.gguf_type_k, "q8_0")
        self.assertEqual(model.gguf_type_v, "q4_0")

    def test_load_settings_defaults_target_inflight_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "test-model": {\n'
                    '        "model_path": "/models/test"\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(path)

        self.assertEqual(settings.engine.models["test-model"].replicas, 1)
        self.assertEqual(settings.engine.models["test-model"].replica_max, 1)
        self.assertEqual(settings.engine.models["test-model"].target_inflight, 1)

    def test_load_settings_reads_replica_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "test-model": {\n'
                    '        "model_path": "/models/test",\n'
                    '        "replicas": 3,\n'
                    '        "replica_max": 4\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            settings = load_settings(path)

        self.assertEqual(settings.engine.models["test-model"].replicas, 3)
        self.assertEqual(settings.engine.models["test-model"].replica_max, 4)

    def test_load_settings_rejects_replicas_above_replica_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                (
                    "{\n"
                    '  "engine": {\n'
                    '    "backend": "stub",\n'
                    '    "models": {\n'
                    '      "test-model": {\n'
                    '        "model_path": "/models/test",\n'
                    '        "replicas": 3,\n'
                    '        "replica_max": 2\n'
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as exc_info:
                load_settings(path)

        self.assertEqual(
            str(exc_info.exception),
            "model 'test-model' has replicas=3 greater than replica_max=2",
        )
