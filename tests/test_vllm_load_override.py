from __future__ import annotations

import importlib.util
import unittest

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import AppSettings
    from app.config import EngineSettings
    from app.config import ModelSettings
    from app.engine.common import _load_constraints_for_backend
    from app.engine.router import ModelRouterEngine


def _router() -> "ModelRouterEngine":
    # _apply_load_override is a pure helper; build the instance without __init__
    return ModelRouterEngine.__new__(ModelRouterEngine)


def _vllm_model(**overrides) -> "ModelSettings":
    return ModelSettings(
        model_path=None,
        backend="vllm",
        vllm_model="Qwen/Qwen2.5-VL-3B-Instruct",
        **overrides,
    )


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class VllmLoadConstraintsTests(unittest.TestCase):
    def test_constraints_expose_the_four_overrides(self) -> None:
        constraints = _load_constraints_for_backend("vllm")
        self.assertEqual(
            set(constraints.keys()),
            {
                "vllm_max_model_len",
                "vllm_kv_cache_dtype",
                "vllm_kv_cache_memory_bytes",
                "vllm_max_pixels",
            },
        )
        self.assertEqual(constraints["vllm_kv_cache_dtype"]["kind"], "enum")
        self.assertIn("fp8", constraints["vllm_kv_cache_dtype"]["allowed_values"])
        self.assertEqual(constraints["vllm_kv_cache_memory_bytes"]["display_unit"], "mib")


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class VllmApplyLoadOverrideTests(unittest.TestCase):
    def test_all_four_overrides_apply(self) -> None:
        result = _router()._apply_load_override(
            _vllm_model(vllm_mm_processor_kwargs=(("max_pixels", 1000),)),
            resolved_backend="vllm",
            load_override={
                "vllm_max_model_len": 8192,
                "vllm_kv_cache_dtype": "fp8",
                "vllm_kv_cache_memory_bytes": 2147483648,
                "vllm_max_pixels": 4014080,
            },
        )
        self.assertTrue(result.enabled)
        self.assertEqual(result.vllm_max_model_len, 8192)
        self.assertEqual(result.vllm_kv_cache_dtype, "fp8")
        self.assertEqual(result.vllm_kv_cache_memory_bytes, 2147483648)
        self.assertEqual(result.vllm_mm_processor_kwargs, (("max_pixels", 4014080),))

    def test_max_pixels_merges_into_existing_mm_kwargs(self) -> None:
        result = _router()._apply_load_override(
            _vllm_model(vllm_mm_processor_kwargs=(("min_pixels", 100), ("max_pixels", 1000))),
            resolved_backend="vllm",
            load_override={"vllm_max_pixels": 5000},
        )
        mm = dict(result.vllm_mm_processor_kwargs)
        self.assertEqual(mm["max_pixels"], 5000)
        self.assertEqual(mm["min_pixels"], 100)

    def test_empty_override_just_enables(self) -> None:
        result = _router()._apply_load_override(
            _vllm_model(),
            resolved_backend="vllm",
            load_override={},
        )
        self.assertTrue(result.enabled)

    def test_unsupported_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _router()._apply_load_override(
                _vllm_model(),
                resolved_backend="vllm",
                load_override={"gguf_n_ctx": 4096},
            )

    def test_invalid_max_model_len_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _router()._apply_load_override(
                _vllm_model(),
                resolved_backend="vllm",
                load_override={"vllm_max_model_len": 0},
            )

    def test_blank_kv_cache_dtype_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _router()._apply_load_override(
                _vllm_model(),
                resolved_backend="vllm",
                load_override={"vllm_kv_cache_dtype": "  "},
            )


if __name__ == "__main__":
    unittest.main()
