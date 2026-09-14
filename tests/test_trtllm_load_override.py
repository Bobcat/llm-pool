from __future__ import annotations

import importlib.util
import unittest

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import ModelSettings
    from app.engine.common import _load_constraints_for_backend
    from app.engine.router import ModelRouterEngine


def _router() -> "ModelRouterEngine":
    return ModelRouterEngine.__new__(ModelRouterEngine)


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class TrtllmLoadOverrideTests(unittest.TestCase):
    def test_constraints_expose_supported_load_settings(self) -> None:
        constraints = _load_constraints_for_backend("trtllm_serve")

        self.assertEqual(
            set(constraints),
            {
                "target_inflight",
                "trtllm_max_seq_len",
                "trtllm_kv_cache_memory_bytes",
                "trtllm_max_num_tokens",
                "trtllm_enable_chunked_prefill",
                "trtllm_kv_cache_dtype",
            },
        )
        self.assertEqual(constraints["trtllm_kv_cache_memory_bytes"]["display_unit"], "mib")
        self.assertEqual(constraints["trtllm_enable_chunked_prefill"]["kind"], "boolean")
        self.assertEqual(
            constraints["trtllm_kv_cache_dtype"]["allowed_values"],
            ["auto", "fp8", "nvfp4"],
        )

    def test_all_overrides_apply(self) -> None:
        result = _router()._apply_load_override(
            ModelSettings(
                model_path=None,
                backend="trtllm_serve",
                trtllm_model="/models/gemma4",
            ),
            resolved_backend="trtllm_serve",
            load_override={
                "trtllm_max_seq_len": 20480,
                "trtllm_kv_cache_memory_bytes": 8589934592,
                "trtllm_max_num_tokens": 8192,
                "trtllm_enable_chunked_prefill": True,
                "trtllm_kv_cache_dtype": " FP8 ",
            },
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.trtllm_max_seq_len, 20480)
        self.assertEqual(result.trtllm_kv_cache_memory_bytes, 8589934592)
        self.assertEqual(result.trtllm_max_num_tokens, 8192)
        self.assertTrue(result.trtllm_enable_chunked_prefill)
        self.assertEqual(result.trtllm_kv_cache_dtype, "fp8")

    def test_invalid_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of: auto, fp8, nvfp4"):
            _router()._apply_load_override(
                ModelSettings(
                    model_path=None,
                    backend="trtllm_serve",
                    trtllm_model="/models/gemma4",
                ),
                resolved_backend="trtllm_serve",
                load_override={"trtllm_kv_cache_dtype": "bf16"},
            )


if __name__ == "__main__":
    unittest.main()
