from __future__ import annotations

import importlib.util
import threading
import unittest
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import DecodingDefaults
    from app.config import ModelSettings
    import app.engine.exllamav3 as exllama_module
    from app.schemas import DecodingParams
    from app.schemas import ResponseRequest


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class ExLlamaV3EngineTests(unittest.TestCase):
    def test_render_prompt_ids_mistral_template_uses_user_turn(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def hf_chat_template(self, messages, **kwargs):
                self.calls.append({"messages": messages, **kwargs})
                return "prompt_ids"

        engine = exllama_module.ExLlamaV3Engine.__new__(exllama_module.ExLlamaV3Engine)
        tokenizer = FakeTokenizer()

        result = engine._render_prompt_ids(
            tokenizer,
            prompt_format="mistral_template",
            system_prompt="Translate to Dutch.",
            user_text="Hello world",
            enable_thinking=None,
        )

        self.assertEqual(result, "prompt_ids")
        self.assertEqual(len(tokenizer.calls), 1)
        call = tokenizer.calls[0]
        self.assertEqual(
            call["messages"],
            [{"role": "user", "content": "Translate to Dutch.\n\nHello world"}],
        )
        self.assertTrue(call["add_generation_prompt"])
        self.assertNotIn("enable_thinking", call)

    def test_render_prompt_ids_gemma4_template_disables_thinking(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def hf_chat_template(self, messages, **kwargs):
                self.calls.append({"messages": messages, **kwargs})
                return "prompt_ids"

        engine = exllama_module.ExLlamaV3Engine.__new__(exllama_module.ExLlamaV3Engine)
        tokenizer = FakeTokenizer()

        result = engine._render_prompt_ids(
            tokenizer,
            prompt_format="gemma4_template",
            system_prompt="Translate to Dutch.",
            user_text="Hello world",
            enable_thinking=False,
        )

        self.assertEqual(result, "prompt_ids")
        self.assertEqual(len(tokenizer.calls), 1)
        call = tokenizer.calls[0]
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": "Translate to Dutch."},
                {"role": "user", "content": "Hello world"},
            ],
        )
        self.assertTrue(call["add_generation_prompt"])
        self.assertFalse(call["enable_thinking"])

    def test_complete_ignores_beam_size_and_logs(self) -> None:
        class FakePromptIds:
            shape = (1, 3)

            def to(self, *, device):
                self.device = device
                return self

        class FakeSampler:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeJob:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeGenerator:
            def __init__(self) -> None:
                self._job = None
                self._remaining = 0

            def enqueue(self, job) -> None:
                self._job = job
                self._remaining = 1

            def num_remaining_jobs(self) -> int:
                return self._remaining

            def iterate(self):
                if self._remaining == 0:
                    return []
                self._remaining = 0
                return [
                    {
                        "job": self._job,
                        "stage": "streaming",
                        "text": "partial",
                        "eos": True,
                        "full_completion": "done",
                        "new_tokens": 2,
                        "time_prefill": 0.01,
                        "time_generate": 0.02,
                    }
                ]

            def clear_queue(self) -> None:
                self._remaining = 0

        class FakeTokenizer:
            eos_token_id = 2
            eos_token_id_list = [2]

        runtime = type("Runtime", (), {})()
        runtime.config = ModelSettings(
            model_path="/models/exl3",
            backend="exllamav3",
            prompt_format="gemma4_template",
        )
        runtime.tokenizer = FakeTokenizer()
        runtime.generator = FakeGenerator()
        runtime.job_class = FakeJob
        runtime.sampler_class = FakeSampler
        runtime.generation_lock = threading.Lock()

        engine = exllama_module.ExLlamaV3Engine.__new__(exllama_module.ExLlamaV3Engine)
        engine.decoding_defaults = DecodingDefaults(
            beam_size=1,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            max_tokens=32,
            stop=[],
        )
        engine._models = {"exl-model": runtime}
        engine._render_prompt_ids = mock.Mock(return_value=FakePromptIds())

        request = ResponseRequest(
            model="exl-model",
            input="hello",
            decoding=DecodingParams(beam_size=7),
        )

        with mock.patch.object(exllama_module.LOGGER, "info") as info_log:
            result = engine.complete(request)

        self.assertEqual(result.text, "done")
        self.assertEqual(result.metrics.engine_output_tokens, 2)
        self.assertGreater(result.metrics.gpu_generate_total_ms, 0.0)
        info_log.assert_called_once()
        self.assertEqual(info_log.call_args[0][1], 7)
        self.assertEqual(info_log.call_args[0][2], "exl-model")
        self.assertIn("<turn|>", runtime.generator._job.kwargs["stop_conditions"])


if __name__ == "__main__":
    unittest.main()
