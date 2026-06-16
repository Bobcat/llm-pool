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


def _vllm_model(*, backend: str = "vllm", **overrides) -> "ModelSettings":
    return ModelSettings(
        model_path=None,
        backend=backend,
        vllm_model="Qwen/Qwen2.5-VL-3B-Instruct",
        **overrides,
    )


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class VllmLoadConstraintsTests(unittest.TestCase):
    def test_constraints_expose_the_live_overrides(self) -> None:
        constraints = _load_constraints_for_backend("vllm")
        self.assertEqual(
            set(constraints.keys()),
            {
                "vllm_max_model_len",
                "vllm_kv_cache_dtype",
                "vllm_kv_cache_memory_bytes",
                "vllm_max_pixels",
                "vllm_speculative_method",
                "vllm_speculative_model",
                "vllm_num_speculative_tokens",
            },
        )
        self.assertEqual(constraints["vllm_kv_cache_dtype"]["kind"], "enum")
        self.assertIn("fp8", constraints["vllm_kv_cache_dtype"]["allowed_values"])
        self.assertEqual(constraints["vllm_kv_cache_memory_bytes"]["display_unit"], "mib")
        self.assertEqual(constraints["vllm_speculative_method"]["kind"], "string_or_null")
        self.assertIn("mtp", constraints["vllm_speculative_method"]["examples"])
        self.assertEqual(constraints["vllm_num_speculative_tokens"]["minimum"], 1)

    def test_vllm_serve_exposes_the_same_overrides(self) -> None:
        self.assertEqual(
            _load_constraints_for_backend("vllm_serve"),
            _load_constraints_for_backend("vllm"),
        )


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class VllmApplyLoadOverrideTests(unittest.TestCase):
    def test_all_overrides_apply(self) -> None:
        result = _router()._apply_load_override(
            _vllm_model(vllm_mm_processor_kwargs=(("max_pixels", 1000),)),
            resolved_backend="vllm",
            load_override={
                "vllm_max_model_len": 8192,
                "vllm_kv_cache_dtype": "fp8",
                "vllm_kv_cache_memory_bytes": 2147483648,
                "vllm_max_pixels": 4014080,
                "vllm_speculative_method": " mtp ",
                "vllm_speculative_model": " google/gemma-4-26B-A4B-it-assistant ",
                "vllm_num_speculative_tokens": 2,
            },
        )
        self.assertTrue(result.enabled)
        self.assertEqual(result.vllm_max_model_len, 8192)
        self.assertEqual(result.vllm_kv_cache_dtype, "fp8")
        self.assertEqual(result.vllm_kv_cache_memory_bytes, 2147483648)
        self.assertEqual(result.vllm_mm_processor_kwargs, (("max_pixels", 4014080),))
        self.assertEqual(result.vllm_speculative_method, "mtp")
        self.assertEqual(result.vllm_speculative_model, "google/gemma-4-26B-A4B-it-assistant")
        self.assertEqual(result.vllm_num_speculative_tokens, 2)

    def test_all_overrides_apply_to_vllm_serve(self) -> None:
        result = _router()._apply_load_override(
            _vllm_model(backend="vllm_serve", vllm_mm_processor_kwargs=(("max_pixels", 1000),)),
            resolved_backend="vllm_serve",
            load_override={
                "vllm_max_model_len": 8192,
                "vllm_kv_cache_dtype": "fp8",
                "vllm_kv_cache_memory_bytes": 2147483648,
                "vllm_max_pixels": 4014080,
                "vllm_speculative_method": "mtp",
                "vllm_speculative_model": "google/gemma-4-26B-A4B-it-assistant",
                "vllm_num_speculative_tokens": 2,
            },
        )
        self.assertTrue(result.enabled)
        self.assertEqual(result.vllm_max_model_len, 8192)
        self.assertEqual(result.vllm_kv_cache_dtype, "fp8")
        self.assertEqual(result.vllm_kv_cache_memory_bytes, 2147483648)
        self.assertEqual(result.vllm_mm_processor_kwargs, (("max_pixels", 4014080),))
        self.assertEqual(result.vllm_speculative_method, "mtp")
        self.assertEqual(result.vllm_speculative_model, "google/gemma-4-26B-A4B-it-assistant")
        self.assertEqual(result.vllm_num_speculative_tokens, 2)

    def test_speculative_method_and_model_accept_null(self) -> None:
        result = _router()._apply_load_override(
            _vllm_model(
                vllm_speculative_method="mtp",
                vllm_speculative_model="google/gemma-4-26B-A4B-it-assistant",
            ),
            resolved_backend="vllm",
            load_override={
                "vllm_speculative_method": None,
                "vllm_speculative_model": None,
            },
        )
        self.assertIsNone(result.vllm_speculative_method)
        self.assertIsNone(result.vllm_speculative_model)

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

    def test_blank_speculative_method_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _router()._apply_load_override(
                _vllm_model(),
                resolved_backend="vllm",
                load_override={"vllm_speculative_method": "  "},
            )

    def test_invalid_num_speculative_tokens_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _router()._apply_load_override(
                _vllm_model(),
                resolved_backend="vllm",
                load_override={"vllm_num_speculative_tokens": 0},
            )


if __name__ == "__main__":
    unittest.main()
