from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import DecodingDefaults
    from app.config import ModelSettings
    import app.engine.llamacpp as llamacpp_module
    from app.schemas import DecodingParams
    from app.schemas import ResponseRequest


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class LlamaCppEngineTests(unittest.TestCase):
    def test_render_prompt_qwen3_without_thinking(self) -> None:
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)

        result = engine._render_prompt(
            prompt_format="qwen3_template",
            system_prompt="System prompt",
            user_text="Hello",
            enable_thinking=False,
        )

        self.assertEqual(
            result,
            "<|im_start|>system\n"
            "System prompt<|im_end|>\n"
            "<|im_start|>user\n"
            "/no_think\n"
            "Hello<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n",
        )

    def test_complete_basic_flow(self) -> None:
        class FakeLlama:
            def tokenize(self, text, *, add_bos, special):
                self.tokenize_text = text
                self.tokenize_kwargs = {"add_bos": add_bos, "special": special}
                return [1, 2, 3, 4, 5]

            def generate(self, tokens, **kwargs):
                self.generate_tokens = tokens
                self.generate_kwargs = kwargs
                yield 100
                yield 200

            def detokenize(self, tokens):
                if tokens == [100]:
                    return b"hello"
                return b"hello world"

            def token_eos(self):
                return 999

        runtime = llamacpp_module.LlamaCppModelRuntime(
            config=ModelSettings(
                model_path="/models/test.gguf",
                backend="gguf",
                prompt_format="generic",
            ),
            llm=FakeLlama(),
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)
        engine.decoding_defaults = DecodingDefaults(
            beam_size=1,
            top_k=40,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            max_tokens=32,
            stop=[],
        )
        engine._models = {"test-gguf": runtime}

        result = engine.complete(ResponseRequest(model="test-gguf", input="Hi"))

        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.metrics.engine_prompt_tokens, 5)
        self.assertEqual(result.metrics.engine_output_tokens, 2)
        self.assertGreaterEqual(result.metrics.gpu_generate_total_ms, 0.0)
        self.assertIsNotNone(result.metrics.engine_tokens_per_second)
        self.assertEqual(runtime.llm.generate_tokens, [1, 2, 3, 4, 5])
        self.assertEqual(runtime.llm.tokenize_kwargs, {"add_bos": False, "special": True})

    def test_complete_stops_on_eos(self) -> None:
        class FakeLlama:
            def tokenize(self, text, *, add_bos, special):
                return [1, 2]

            def generate(self, tokens, **kwargs):
                yield 10
                yield 999
                yield 20

            def detokenize(self, tokens):
                return b"stopped"

            def token_eos(self):
                return 999

        runtime = llamacpp_module.LlamaCppModelRuntime(
            config=ModelSettings(model_path="/models/t.gguf", backend="gguf"),
            llm=FakeLlama(),
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)
        engine.decoding_defaults = DecodingDefaults(max_tokens=100)
        engine._models = {"m": runtime}

        result = engine.complete(ResponseRequest(model="m", input="x"))

        self.assertEqual(result.metrics.engine_output_tokens, 2)

    def test_complete_stops_on_max_tokens(self) -> None:
        class FakeLlama:
            def tokenize(self, text, *, add_bos, special):
                return [1]

            def generate(self, tokens, **kwargs):
                for i in range(100):
                    yield i

            def detokenize(self, tokens):
                return b"output"

            def token_eos(self):
                return 999

        runtime = llamacpp_module.LlamaCppModelRuntime(
            config=ModelSettings(model_path="/models/t.gguf", backend="gguf"),
            llm=FakeLlama(),
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)
        engine.decoding_defaults = DecodingDefaults(max_tokens=5)
        engine._models = {"m": runtime}

        result = engine.complete(ResponseRequest(model="m", input="x"))

        self.assertEqual(result.metrics.engine_output_tokens, 5)

    def test_complete_stops_on_stop_string(self) -> None:
        class FakeLlama:
            def tokenize(self, text, *, add_bos, special):
                return [1, 2]

            def generate(self, tokens, **kwargs):
                yield 10
                yield 20
                yield 30

            def detokenize(self, tokens):
                if tokens == [10]:
                    return b"hello"
                if tokens == [10, 20]:
                    return b"hello</stop>"
                return b"hello</stop>ignored"

            def token_eos(self):
                return 999

        runtime = llamacpp_module.LlamaCppModelRuntime(
            config=ModelSettings(model_path="/models/t.gguf", backend="gguf"),
            llm=FakeLlama(),
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)
        engine.decoding_defaults = DecodingDefaults(stop=["</stop>"])
        engine._models = {"m": runtime}

        result = engine.complete(ResponseRequest(model="m", input="x"))

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.metrics.engine_output_tokens, 2)

    def test_complete_ignores_beam_size_and_logs(self) -> None:
        class FakeLlama:
            def tokenize(self, text, *, add_bos, special):
                return [1]

            def generate(self, tokens, **kwargs):
                yield 10

            def detokenize(self, tokens):
                return b"done"

            def token_eos(self):
                return 999

        runtime = llamacpp_module.LlamaCppModelRuntime(
            config=ModelSettings(model_path="/models/t.gguf", backend="gguf"),
            llm=FakeLlama(),
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)
        engine.decoding_defaults = DecodingDefaults()
        engine._models = {"gguf-model": runtime}
        request = ResponseRequest(
            model="gguf-model",
            input="hello",
            decoding=DecodingParams(beam_size=7),
        )

        with mock.patch.object(llamacpp_module.LOGGER, "info") as info_log:
            result = engine.complete(request)

        self.assertEqual(result.text, "done")
        info_log.assert_called_once()
        self.assertEqual(info_log.call_args[0][1], 7)
        self.assertEqual(info_log.call_args[0][2], "gguf-model")

    def test_complete_gemma4_uses_native_chat_completion(self) -> None:
        class FakeLlama:
            def create_chat_completion(self, **kwargs):
                self.chat_kwargs = kwargs
                return {
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 1},
                }

        runtime = llamacpp_module.LlamaCppModelRuntime(
            config=ModelSettings(
                model_path="/models/gemma4.gguf",
                backend="gguf",
                prompt_format="gemma4_template",
            ),
            llm=FakeLlama(),
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)
        engine.decoding_defaults = DecodingDefaults(
            top_k=12,
            top_p=0.8,
            temperature=0.3,
            repetition_penalty=1.05,
            max_tokens=9,
            stop=["</stop>"],
        )
        engine._models = {"gemma4-gguf": runtime}

        result = engine.complete(ResponseRequest(model="gemma4-gguf", input="Reply with OK"))

        self.assertEqual(result.text, "OK")
        self.assertEqual(result.metrics.engine_prompt_tokens, 12)
        self.assertEqual(result.metrics.engine_output_tokens, 1)
        self.assertEqual(
            runtime.llm.chat_kwargs["messages"],
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Return only the response.",
                },
                {"role": "user", "content": "Reply with OK"},
            ],
        )
        self.assertEqual(runtime.llm.chat_kwargs["top_k"], 12)
        self.assertEqual(runtime.llm.chat_kwargs["top_p"], 0.8)
        self.assertEqual(runtime.llm.chat_kwargs["temperature"], 0.3)
        self.assertEqual(runtime.llm.chat_kwargs["repeat_penalty"], 1.05)
        self.assertEqual(runtime.llm.chat_kwargs["max_tokens"], 9)
        self.assertIn("<end_of_turn>", runtime.llm.chat_kwargs["stop"])
        self.assertIn("</stop>", runtime.llm.chat_kwargs["stop"])

    def test_build_runtime_passes_gguf_cache_types(self) -> None:
        captured: dict[str, object] = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = FakeLlama
        fake_module.GGML_TYPE_Q8_0 = 17
        fake_module.GGML_TYPE_Q4_0 = 18

        settings = ModelSettings(
            model_path="/models/test.gguf",
            backend="gguf",
            gguf_n_ctx=8192,
            gguf_flash_attn="off",
            gguf_type_k="q8_0",
            gguf_type_v="q4_0",
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)

        with mock.patch.dict(sys.modules, {"llama_cpp": fake_module}):
            runtime = engine._build_runtime(settings)

        self.assertIs(runtime.llm.__class__, FakeLlama)
        self.assertEqual(captured["model_path"], "/models/test.gguf")
        self.assertEqual(captured["n_ctx"], 8192)
        self.assertFalse(captured["flash_attn"])
        self.assertEqual(captured["type_k"], 17)
        self.assertEqual(captured["type_v"], 18)

    def test_build_runtime_maps_auto_flash_attn_to_low_level_auto_constant(self) -> None:
        captured: dict[str, object] = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                captured["disabled_constant_during_init"] = fake_low_level.LLAMA_FLASH_ATTN_TYPE_DISABLED

        fake_package = types.ModuleType("llama_cpp")
        fake_low_level = types.ModuleType("llama_cpp.llama_cpp")
        fake_package.Llama = FakeLlama
        fake_package.llama_cpp = fake_low_level
        fake_package.LLAMA_FLASH_ATTN_TYPE_AUTO = -1
        fake_package.LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
        fake_low_level.LLAMA_FLASH_ATTN_TYPE_AUTO = -1
        fake_low_level.LLAMA_FLASH_ATTN_TYPE_DISABLED = 0

        settings = ModelSettings(
            model_path="/models/test.gguf",
            backend="gguf",
            gguf_flash_attn="auto",
        )
        engine = llamacpp_module.LlamaCppEngine.__new__(llamacpp_module.LlamaCppEngine)

        with mock.patch.dict(
            sys.modules,
            {
                "llama_cpp": fake_package,
                "llama_cpp.llama_cpp": fake_low_level,
            },
        ):
            runtime = engine._build_runtime(settings)

        self.assertIs(runtime.llm.__class__, FakeLlama)
        self.assertFalse(captured["flash_attn"])
        self.assertEqual(captured["disabled_constant_during_init"], -1)
        self.assertEqual(fake_low_level.LLAMA_FLASH_ATTN_TYPE_DISABLED, 0)


if __name__ == "__main__":
    unittest.main()
