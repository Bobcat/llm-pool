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
class SglangLoadOverrideTests(unittest.TestCase):
    def test_constraints_expose_supported_load_settings(self) -> None:
        constraints = _load_constraints_for_backend("sglang_serve")

        self.assertEqual(
            set(constraints),
            {
                "target_inflight",
                "sglang_context_length",
                "sglang_mem_fraction_static",
                "sglang_max_total_tokens",
                "sglang_chunked_prefill_size",
                "sglang_kv_cache_dtype",
                "sglang_speculative_algorithm",
                "sglang_speculative_draft_model",
                "sglang_speculative_num_steps",
                "sglang_speculative_num_draft_tokens",
                "sglang_speculative_eagle_topk",
            },
        )
        self.assertEqual(constraints["sglang_max_total_tokens"]["unit"], "tokens")
        self.assertEqual(constraints["sglang_max_total_tokens"]["minimum"], 256)
        self.assertIn(
            "fp8_e4m3",
            constraints["sglang_kv_cache_dtype"]["allowed_values"],
        )

    def test_all_overrides_apply(self) -> None:
        result = _router()._apply_load_override(
            ModelSettings(
                model_path=None,
                backend="sglang_serve",
                sglang_model="/models/gemma4",
            ),
            resolved_backend="sglang_serve",
            load_override={
                "sglang_context_length": 20480,
                "sglang_mem_fraction_static": 0.3,
                "sglang_max_total_tokens": 20480,
                "sglang_chunked_prefill_size": 8192,
                "sglang_kv_cache_dtype": " FP8_E4M3 ",
                "sglang_speculative_algorithm": "NEXTN",
                "sglang_speculative_draft_model": "assistant",
                "sglang_speculative_num_steps": 5,
                "sglang_speculative_num_draft_tokens": 6,
                "sglang_speculative_eagle_topk": 1,
            },
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.sglang_context_length, 20480)
        self.assertEqual(result.sglang_mem_fraction_static, 0.3)
        self.assertEqual(result.sglang_max_total_tokens, 20480)
        self.assertEqual(result.sglang_chunked_prefill_size, 8192)
        self.assertEqual(result.sglang_kv_cache_dtype, "fp8_e4m3")
        self.assertEqual(result.sglang_speculative_algorithm, "NEXTN")
        self.assertEqual(result.sglang_speculative_draft_model, "assistant")
        self.assertEqual(result.sglang_speculative_num_steps, 5)
        self.assertEqual(result.sglang_speculative_num_draft_tokens, 6)
        self.assertEqual(result.sglang_speculative_eagle_topk, 1)

    def test_invalid_chunked_prefill_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be -1 or a positive integer"):
            _router()._apply_load_override(
                ModelSettings(
                    model_path=None,
                    backend="sglang_serve",
                    sglang_model="/models/gemma4",
                ),
                resolved_backend="sglang_serve",
                load_override={"sglang_chunked_prefill_size": 0},
            )

    def test_null_algorithm_disables_configured_speculative_decoding(self) -> None:
        result = _router()._apply_load_override(
            ModelSettings(
                model_path=None,
                backend="sglang_serve",
                sglang_model="/models/gemma4",
                sglang_speculative_algorithm="NEXTN",
                sglang_speculative_draft_model="assistant",
            ),
            resolved_backend="sglang_serve",
            load_override={"sglang_speculative_algorithm": None},
        )

        self.assertIsNone(result.sglang_speculative_algorithm)
        self.assertEqual(result.sglang_speculative_draft_model, "assistant")

    def test_topk_one_derives_draft_tokens_from_overridden_steps(self) -> None:
        result = _router()._apply_load_override(
            ModelSettings(
                model_path=None,
                backend="sglang_serve",
                sglang_model="/models/gemma4",
                sglang_speculative_algorithm="NEXTN",
                sglang_speculative_num_steps=5,
                sglang_speculative_num_draft_tokens=6,
                sglang_speculative_eagle_topk=1,
            ),
            resolved_backend="sglang_serve",
            load_override={"sglang_speculative_num_steps": 3},
        )

        self.assertEqual(result.sglang_speculative_num_steps, 3)
        self.assertEqual(result.sglang_speculative_num_draft_tokens, 4)

    def test_topk_one_derives_draft_tokens_for_unrelated_override(self) -> None:
        result = _router()._apply_load_override(
            ModelSettings(
                model_path=None,
                backend="sglang_serve",
                sglang_model="/models/gemma4",
                sglang_speculative_algorithm="NEXTN",
                sglang_speculative_num_steps=3,
                sglang_speculative_num_draft_tokens=6,
                sglang_speculative_eagle_topk=1,
            ),
            resolved_backend="sglang_serve",
            load_override={"sglang_kv_cache_dtype": "fp8_e4m3"},
        )

        self.assertEqual(result.sglang_speculative_num_steps, 3)
        self.assertEqual(result.sglang_speculative_num_draft_tokens, 4)

    def test_topk_one_rejects_conflicting_explicit_draft_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal"):
            _router()._apply_load_override(
                ModelSettings(
                    model_path=None,
                    backend="sglang_serve",
                    sglang_model="/models/gemma4",
                    sglang_speculative_algorithm="NEXTN",
                    sglang_speculative_num_steps=5,
                    sglang_speculative_num_draft_tokens=6,
                    sglang_speculative_eagle_topk=1,
                ),
                resolved_backend="sglang_serve",
                load_override={"sglang_speculative_num_draft_tokens": 4},
            )


if __name__ == "__main__":
    unittest.main()
