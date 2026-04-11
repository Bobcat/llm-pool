from __future__ import annotations

import importlib.util
import threading
import unittest
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    import app.engine as engine_module
    from app.config import AppSettings
    from app.config import DecodingDefaults
    from app.config import EngineSettings
    from app.config import ModelSettings
    from app.config import ServiceSettings
    from app.engine import Ct2Engine
    from app.engine import ModelRouterEngine
    from app.schemas import DecodingParams
    from app.schemas import EngineResult
    from app.schemas import ResponseRequest


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text, *, add_special_tokens):
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return {"input_ids": [11, 22, 33]}

    def convert_ids_to_tokens(self, input_ids):
        return [f"tok-{value}" for value in input_ids]


class FakeTokenizerWithVocab:
    def __init__(self) -> None:
        self.unk_token_id = 0
        self.eos_token = "</s>"
        self._vocab = {"</s>": 1, "<s>": 2}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


class FakeMistralTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.tokenize_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "return_tensors": return_tensors,
            }
        )
        return "<s>[INST] Translate to Dutch.\n\nHello [/INST]"

    def __call__(self, text, *, add_special_tokens):
        self.tokenize_calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return {"input_ids": [7, 8]}

    def convert_ids_to_tokens(self, ids):
        return [f"tok-{item}" for item in ids]


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class EngineTests(unittest.TestCase):
    def test_render_qwen3_prompt_tokens_uses_non_thinking_prompt_prefix(self) -> None:
        engine = Ct2Engine.__new__(Ct2Engine)
        tokenizer = FakeQwenTokenizer()

        tokens = engine._render_qwen3_prompt_tokens(
            tokenizer,
            system_prompt="System prompt",
            user_text="User text",
        )

        self.assertEqual(tokens, ["tok-11", "tok-22", "tok-33"])
        self.assertEqual(len(tokenizer.calls), 1)
        call = tokenizer.calls[0]
        self.assertEqual(
            call["text"],
            "<|im_start|>system\n"
            "System prompt<|im_end|>\n"
            "<|im_start|>user\n"
            "/no_think\n"
            "User text<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
        )
        self.assertFalse(call["add_special_tokens"])

    def test_resolve_decoding_prefers_payload_then_settings_then_defaults(self) -> None:
        engine = Ct2Engine.__new__(Ct2Engine)
        engine.decoding_defaults = DecodingDefaults(
            beam_size=2,
            top_k=5,
            top_p=0.9,
            temperature=0.3,
            repetition_penalty=1.2,
            max_tokens=333,
            stop=["</s>"],
        )

        resolved = engine._resolve_decoding(
            DecodingParams(
                beam_size=1,
                top_k=None,
                top_p=None,
                temperature=0.1,
                repetition_penalty=None,
                max_tokens=None,
                stop=None,
            )
        )

        self.assertEqual(resolved.beam_size, 1)
        self.assertEqual(resolved.top_k, 5)
        self.assertEqual(resolved.top_p, 0.9)
        self.assertEqual(resolved.temperature, 0.1)
        self.assertEqual(resolved.repetition_penalty, 1.2)
        self.assertEqual(resolved.max_tokens, 333)
        self.assertEqual(resolved.stop, ["</s>"])

    def test_disabled_models_are_not_loaded(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                default_model="enabled-model",
                models={
                    "enabled-model": ModelSettings(model_path="/models/enabled", enabled=True),
                    "disabled-model": ModelSettings(model_path="/models/disabled", enabled=False),
                },
            ),
        )
        engine = Ct2Engine.__new__(Ct2Engine)
        engine.default_model = settings.engine.default_model
        engine.decoding_defaults = settings.engine.decoding
        engine._models = {}
        seen: list[str] = []

        def fake_build_runtime(model_settings):
            seen.append(model_settings.model_path)
            return object()

        engine._build_runtime = fake_build_runtime  # type: ignore[method-assign]
        for model_name, model_settings in settings.engine.models.items():
            if not model_settings.enabled:
                continue
            engine._models[model_name] = engine._build_runtime(model_settings)

        self.assertEqual(seen, ["/models/enabled"])
        self.assertIn("enabled-model", engine._models)
        self.assertNotIn("disabled-model", engine._models)

    def test_resolve_end_token_falls_back_when_im_end_not_in_vocab(self) -> None:
        engine = Ct2Engine.__new__(Ct2Engine)
        tokenizer = FakeTokenizerWithVocab()

        end_token = engine._resolve_end_token(tokenizer, ["<|im_end|>"])

        self.assertEqual(end_token, "</s>")

    def test_render_mistral_prompt_tokens_uses_chat_template_user_turn(self) -> None:
        engine = Ct2Engine.__new__(Ct2Engine)
        tokenizer = FakeMistralTokenizer()

        tokens = engine._render_mistral_prompt_tokens(
            tokenizer,
            system_prompt="Translate to Dutch.",
            user_text="Hello",
        )

        self.assertEqual(tokens, ["tok-7", "tok-8"])
        self.assertEqual(len(tokenizer.calls), 1)
        call = tokenizer.calls[0]
        self.assertEqual(
            call["messages"],
            [{"role": "user", "content": "Translate to Dutch.\n\nHello"}],
        )
        self.assertFalse(call["tokenize"])
        self.assertTrue(call["add_generation_prompt"])
        self.assertIsNone(call["return_tensors"])
        self.assertEqual(len(tokenizer.tokenize_calls), 1)
        self.assertEqual(
            tokenizer.tokenize_calls[0]["text"],
            "<s>[INST] Translate to Dutch.\n\nHello [/INST]",
        )
        self.assertFalse(tokenizer.tokenize_calls[0]["add_special_tokens"])

    def test_model_router_engine_dispatches_by_model_backend(self) -> None:
        settings = AppSettings(
            service=ServiceSettings(),
            engine=EngineSettings(
                backend="ct2",
                default_model="ct2-model",
                models={
                    "ct2-model": ModelSettings(model_path="/models/ct2"),
                    "exl-model": ModelSettings(model_path="/models/exl3", backend="exllamav3"),
                },
            ),
        )

        class FakeCt2Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"ct2:{request.model}")

        class FakeExLlamaV3Engine:
            def __init__(self, scoped_settings):
                self._models = {name: object() for name in scoped_settings.engine.models}

            def complete(self, request: ResponseRequest) -> EngineResult:
                return EngineResult(text=f"exl3:{request.model}")

        with (
            mock.patch.object(engine_module, "Ct2Engine", FakeCt2Engine),
            mock.patch.object(engine_module, "ExLlamaV3Engine", FakeExLlamaV3Engine),
        ):
            engine = ModelRouterEngine(settings)
            ct2_result = engine.complete(ResponseRequest(model="ct2-model", input="hello"))
            exl_result = engine.complete(ResponseRequest(model="exl-model", input="hello"))

        self.assertEqual(ct2_result.text, "ct2:ct2-model")
        self.assertEqual(exl_result.text, "exl3:exl-model")
        self.assertEqual(sorted(engine._models.keys()), ["ct2-model", "exl-model"])
        self.assertEqual(engine.default_model, "ct2-model")

    def test_exllamav3_complete_ignores_beam_size_and_logs(self) -> None:
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
        runtime.config = ModelSettings(model_path="/models/exl3", backend="exllamav3", prompt_format="generic")
        runtime.tokenizer = FakeTokenizer()
        runtime.generator = FakeGenerator()
        runtime.job_class = FakeJob
        runtime.sampler_class = FakeSampler
        runtime.generation_lock = threading.Lock()

        engine = engine_module.ExLlamaV3Engine.__new__(engine_module.ExLlamaV3Engine)
        engine.decoding_defaults = DecodingDefaults(
            beam_size=1,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            max_tokens=32,
            stop=["</s>"],
        )
        engine._models = {"exl-model": runtime}
        engine._render_prompt_ids = mock.Mock(return_value=FakePromptIds())

        request = ResponseRequest(
            model="exl-model",
            input="hello",
            decoding=DecodingParams(beam_size=7),
        )

        with mock.patch.object(engine_module.LOGGER, "info") as info_log:
            result = engine.complete(request)

        self.assertEqual(result.text, "done")
        self.assertEqual(result.metrics.engine_output_tokens, 2)
        self.assertGreater(result.metrics.gpu_generate_total_ms, 0.0)
        info_log.assert_called_once()
        self.assertEqual(info_log.call_args[0][1], 7)
        self.assertEqual(info_log.call_args[0][2], "exl-model")

if __name__ == "__main__":
    unittest.main()
