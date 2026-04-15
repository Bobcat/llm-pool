from __future__ import annotations

import importlib.util
import unittest

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import DecodingDefaults
    from app.config import ModelSettings
    from app.engine import Ct2Engine
    from app.engine import Ct2ModelRuntime
    from app.schemas import DecodingParams
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
        if isinstance(token, list):
            return [self._vocab.get(item, self.unk_token_id) for item in token]
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


class FakeQwenCompleteTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.eos_token = "</s>"
        self.unk_token_id = 0
        self._vocab = {
            "<think>": 1,
            "</think>": 2,
            "</s>": 3,
            "tok-a": 4,
            "tok-b": 5,
        }

    def __call__(self, text, *, add_special_tokens):
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return {"input_ids": [11, 22, 33]}

    def convert_ids_to_tokens(self, input_ids):
        return [f"tok-{value}" for value in input_ids]

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self._vocab.get(item, self.unk_token_id) for item in token]
        return self._vocab.get(token, self.unk_token_id)

    def decode(self, token_ids, skip_special_tokens=True):
        return "done"


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class Ct2EngineTests(unittest.TestCase):
    def test_render_qwen3_prompt_tokens_uses_non_thinking_prompt_prefix(self) -> None:
        engine = Ct2Engine.__new__(Ct2Engine)
        tokenizer = FakeQwenTokenizer()

        tokens = engine._render_qwen3_prompt_tokens(
            tokenizer,
            system_prompt="System prompt",
            user_text="User text",
            enable_thinking=False,
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

    def test_render_qwen3_prompt_tokens_can_leave_thinking_enabled(self) -> None:
        engine = Ct2Engine.__new__(Ct2Engine)
        tokenizer = FakeQwenTokenizer()

        tokens = engine._render_qwen3_prompt_tokens(
            tokenizer,
            system_prompt="System prompt",
            user_text="User text",
            enable_thinking=True,
        )

        self.assertEqual(tokens, ["tok-11", "tok-22", "tok-33"])
        self.assertEqual(len(tokenizer.calls), 1)
        call = tokenizer.calls[0]
        self.assertEqual(
            call["text"],
            "<|im_start|>system\n"
            "System prompt<|im_end|>\n"
            "<|im_start|>user\n"
            "User text<|im_end|>\n"
            "<|im_start|>assistant\n",
        )
        self.assertFalse(call["add_special_tokens"])

    def test_qwen3_complete_does_not_suppress_thinking_when_enabled(self) -> None:
        class FakeResult:
            def __init__(self) -> None:
                self.sequences = [[101, 102]]

        class FakeGenerator:
            def __init__(self) -> None:
                self.kwargs = None

            def generate_batch(self, prompt_batches, **kwargs):
                self.kwargs = kwargs
                return [FakeResult()]

        runtime = Ct2ModelRuntime(
            config=ModelSettings(
                model_path="/models/qwen3",
                prompt_format="qwen3_template",
                enable_thinking=True,
            ),
            generator=FakeGenerator(),
            tokenizer=FakeQwenCompleteTokenizer(),
        )
        engine = Ct2Engine.__new__(Ct2Engine)
        engine.decoding_defaults = DecodingDefaults(
            beam_size=1,
            top_k=1,
            top_p=1.0,
            temperature=0.1,
            repetition_penalty=1.0,
            max_tokens=16,
            stop=["</s>"],
        )
        engine._models = {"qwen3": runtime}

        result = engine.complete(ResponseRequest(model="qwen3", input="Hello"))

        self.assertEqual(result.text, "done")
        self.assertIsNone(runtime.generator.kwargs["suppress_sequences"])

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
        engine = Ct2Engine.__new__(Ct2Engine)
        engine.decoding_defaults = DecodingDefaults()
        engine._models = {}
        seen: list[str] = []

        def fake_build_runtime(model_settings):
            seen.append(model_settings.model_path)
            return object()

        engine._build_runtime = fake_build_runtime  # type: ignore[method-assign]
        models = {
            "enabled-model": ModelSettings(model_path="/models/enabled", enabled=True),
            "disabled-model": ModelSettings(model_path="/models/disabled", enabled=False),
        }
        for model_name, model_settings in models.items():
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

    def test_generic_complete_uses_native_im_end_stop_without_config_default(self) -> None:
        class FakeResult:
            def __init__(self) -> None:
                self.sequences = [[101, 102]]

        class FakeGenerator:
            def __init__(self) -> None:
                self.kwargs = None

            def generate_batch(self, prompt_batches, **kwargs):
                self.kwargs = kwargs
                return [FakeResult()]

        class FakeTokenizer(FakeTokenizerWithVocab):
            def __init__(self) -> None:
                super().__init__()
                self._vocab["<|im_end|>"] = 6

            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": [11, 22]}

            def convert_ids_to_tokens(self, input_ids):
                return [f"tok-{value}" for value in input_ids]

            def decode(self, token_ids, skip_special_tokens=True):
                return "done"

        runtime = Ct2ModelRuntime(
            config=ModelSettings(model_path="/models/generic", prompt_format="generic"),
            generator=FakeGenerator(),
            tokenizer=FakeTokenizer(),
        )
        engine = Ct2Engine.__new__(Ct2Engine)
        engine.decoding_defaults = DecodingDefaults(
            beam_size=1,
            top_k=1,
            top_p=1.0,
            temperature=0.1,
            repetition_penalty=1.0,
            max_tokens=16,
            stop=[],
        )
        engine._models = {"generic": runtime}

        result = engine.complete(ResponseRequest(model="generic", input="Hello"))

        self.assertEqual(result.text, "done")
        self.assertEqual(runtime.generator.kwargs["end_token"], "<|im_end|>")


if __name__ == "__main__":
    unittest.main()
