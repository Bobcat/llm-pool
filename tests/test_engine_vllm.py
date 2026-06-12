"""Unit tests for the vllm backend.

These tests mock the vllm engine so they don't require a real GPU or model
load. End-to-end inference is verified separately via the smoke-test script.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import threading
import unittest
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None

if HAS_PYDANTIC:
    from app.config import AppSettings
    from app.config import EngineSettings
    from app.config import ModelSettings
    from app.engine import BackendExecutionError
    from app.engine.vllm import VllmEngine
    from app.engine.vllm import VllmModelRuntime
    from app.schemas import ImageContent
    from app.schemas import ImageUrlSpec
    from app.schemas import Message
    from app.schemas import ResponseRequest
    from app.schemas import TextContent


def _make_settings(**overrides) -> "ModelSettings":
    return ModelSettings(
        model_path=overrides.pop("model_path", "/tmp/fake-model"),
        backend="vllm",
        prompt_format=overrides.pop("prompt_format", "generic"),
        modalities=overrides.pop("modalities", ("text", "image")),
        **overrides,
    )


def _png_bytes() -> bytes:
    if not HAS_PIL:
        return b""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(HAS_PYDANTIC and HAS_PIL, "pydantic or PIL not installed")
class VllmTextExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Build a VllmEngine instance without calling __init__ (no engine load)
        self.engine = VllmEngine.__new__(VllmEngine)

    def test_string_input_returns_text_and_empty_images(self) -> None:
        request = ResponseRequest(model="m", input="Hello")
        texts, images = self.engine._extract_text_and_images(request)
        self.assertEqual(texts, ["Hello"])
        self.assertEqual(images, [])

    def test_text_only_content_list_returns_joined_text(self) -> None:
        request = ResponseRequest(
            model="m",
            input=[TextContent(text="Hello "), TextContent(text="world")],
        )
        texts, images = self.engine._extract_text_and_images(request)
        self.assertEqual(texts, ["Hello ", "world"])
        self.assertEqual(images, [])

    def test_mixed_content_loads_image_and_keeps_text(self) -> None:
        png = _png_bytes()
        b64 = base64.b64encode(png).decode("ascii")
        request = ResponseRequest(
            model="m",
            input=[
                TextContent(text="Describe:"),
                ImageContent(image_url=ImageUrlSpec(url=f"data:image/png;base64,{b64}")),
            ],
        )
        texts, images = self.engine._extract_text_and_images(request)
        self.assertEqual(texts, ["Describe:"])
        self.assertEqual(len(images), 1)
        # Image is now a PIL image
        self.assertTrue(hasattr(images[0], "size"))
        self.assertEqual(images[0].size, (1, 1))


@unittest.skipUnless(HAS_PYDANTIC and HAS_PIL, "pydantic or PIL not installed")
class VllmImageDecodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = VllmEngine.__new__(VllmEngine)

    def test_data_url_base64_decodes(self) -> None:
        png = _png_bytes()
        b64 = base64.b64encode(png).decode("ascii")
        item = ImageContent(image_url=ImageUrlSpec(url=f"data:image/png;base64,{b64}"))
        image = self.engine._load_image(item)
        self.assertEqual(image.size, (1, 1))

    def test_data_url_without_base64_marker_is_rejected(self) -> None:
        item = ImageContent(image_url=ImageUrlSpec(url="data:image/png,abc"))
        with self.assertRaises(BackendExecutionError) as exc_info:
            self.engine._load_image(item)
        self.assertEqual(exc_info.exception.code, "image_data_url_invalid")

    def test_unsupported_scheme_is_rejected(self) -> None:
        item = ImageContent(image_url=ImageUrlSpec(url="ftp://example.com/x.png"))
        with self.assertRaises(BackendExecutionError) as exc_info:
            self.engine._load_image(item)
        self.assertEqual(exc_info.exception.code, "image_url_scheme_unsupported")

    def test_empty_url_is_rejected(self) -> None:
        item = ImageContent(image_url=ImageUrlSpec(url="   "))
        with self.assertRaises(BackendExecutionError) as exc_info:
            self.engine._load_image(item)
        self.assertEqual(exc_info.exception.code, "image_url_empty")


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class VllmPromptRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = VllmEngine.__new__(VllmEngine)

    def _make_runtime(self, captured: dict, **settings_overrides) -> "VllmModelRuntime":
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return "<rendered prompt>"

        return VllmModelRuntime(
            config=_make_settings(**settings_overrides),
            engine=object(),
            tokenizer=FakeTokenizer(),
            loop=asyncio.new_event_loop(),
            loop_thread=threading.Thread(target=lambda: None),
        )

    def test_text_only_renders_string_user_content(self) -> None:
        captured: dict = {}
        runtime = self._make_runtime(captured)
        result = self.engine._render_prompt(
            runtime=runtime,
            system_prompt="You are X.",
            user_text="Hello",
            has_images=False,
        )
        self.assertEqual(result, "<rendered prompt>")
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["messages"][0]["content"], "You are X.")
        self.assertEqual(captured["messages"][1]["role"], "user")
        self.assertEqual(captured["messages"][1]["content"], "Hello")
        self.assertTrue(captured["kwargs"]["add_generation_prompt"])
        self.assertFalse(captured["kwargs"]["tokenize"])

    def test_text_only_can_pass_enable_thinking_to_chat_template(self) -> None:
        captured: dict = {}
        runtime = self._make_runtime(captured)
        self.engine._render_prompt(
            runtime=runtime,
            system_prompt="You are X.",
            user_text="Hello",
            has_images=False,
            enable_thinking=True,
        )
        self.assertTrue(captured["kwargs"]["enable_thinking"])

    def test_request_enable_thinking_uses_gemma4_request_override(self) -> None:
        captured: dict = {}
        runtime = self._make_runtime(
            captured,
            prompt_format="gemma4_template",
            enable_thinking=False,
        )

        enabled = self.engine._request_enable_thinking(
            runtime,
            ResponseRequest(model="m", input="Hello", thinking="enabled"),
        )

        self.assertTrue(enabled)

    def test_request_enable_thinking_rejects_generic_override(self) -> None:
        captured: dict = {}
        runtime = self._make_runtime(captured)

        with self.assertRaises(BackendExecutionError) as exc_info:
            self.engine._request_enable_thinking(
                runtime,
                ResponseRequest(model="m", input="Hello", thinking="enabled"),
            )

        self.assertEqual(exc_info.exception.code, "thinking_unsupported")

    def test_with_image_uses_content_list_with_placeholder(self) -> None:
        captured: dict = {}
        runtime = self._make_runtime(captured)
        self.engine._render_prompt(
            runtime=runtime,
            system_prompt="You are X.",
            user_text="Describe",
            has_images=True,
        )
        user_content = captured["messages"][1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "image")
        self.assertEqual(user_content[1]["type"], "text")
        self.assertEqual(user_content[1]["text"], "Describe")

    def test_chat_template_failure_raises_backend_execution_error(self) -> None:
        class BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("template error")

        runtime = VllmModelRuntime(
            config=_make_settings(),
            engine=object(),
            tokenizer=BrokenTokenizer(),
            loop=asyncio.new_event_loop(),
            loop_thread=threading.Thread(target=lambda: None),
        )
        with self.assertRaises(BackendExecutionError) as exc_info:
            self.engine._render_prompt(
                runtime=runtime,
                system_prompt="X",
                user_text="Y",
                has_images=False,
            )
        self.assertEqual(exc_info.exception.code, "chat_template_render_failed")


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class VllmConversationRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = VllmEngine.__new__(VllmEngine)

    def _make_runtime(self, captured: dict) -> "VllmModelRuntime":
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return "<rendered conversation>"

        return VllmModelRuntime(
            config=_make_settings(),
            engine=object(),
            tokenizer=FakeTokenizer(),
            loop=asyncio.new_event_loop(),
            loop_thread=threading.Thread(target=lambda: None),
        )

    def test_multi_turn_prepends_system_and_preserves_history(self) -> None:
        captured: dict = {}
        runtime = self._make_runtime(captured)
        result = self.engine._render_conversation_prompt(
            runtime=runtime,
            system_prompt="You are X.",
            messages=[
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello!"),
                Message(role="user", content="And now?"),
            ],
        )
        self.assertEqual(result, "<rendered conversation>")
        self.assertEqual(
            [(m["role"], m["content"]) for m in captured["messages"]],
            [
                ("system", "You are X."),
                ("user", "Hi"),
                ("assistant", "Hello!"),
                ("user", "And now?"),
            ],
        )
        self.assertTrue(captured["kwargs"]["add_generation_prompt"])
        self.assertFalse(captured["kwargs"]["tokenize"])

    def test_text_content_list_is_flattened(self) -> None:
        captured: dict = {}
        runtime = self._make_runtime(captured)
        self.engine._render_conversation_prompt(
            runtime=runtime,
            system_prompt="X",
            messages=[
                Message(
                    role="user",
                    content=[TextContent(text="foo "), TextContent(text="bar")],
                )
            ],
        )
        self.assertEqual(captured["messages"][1]["content"], "foo bar")

    def test_text_renderer_rejects_image_turns(self) -> None:
        # Image-bearing conversations are routed to the multimodal path; the
        # text renderer guards defensively if called directly with an image.
        captured: dict = {}
        runtime = self._make_runtime(captured)
        message = Message(
            role="user",
            content=[ImageContent(image_url=ImageUrlSpec(url="data:image/png;base64,AA"))],
        )
        with self.assertRaises(BackendExecutionError) as exc_info:
            self.engine._render_conversation_prompt(
                runtime=runtime,
                system_prompt="X",
                messages=[message],
            )
        self.assertEqual(exc_info.exception.code, "multi_turn_image_unsupported")

    def test_conversation_has_images_detects_image_turns(self) -> None:
        text_only = [
            Message(role="user", content="hi"),
            Message(role="assistant", content=[TextContent(text="hello")]),
        ]
        with_image = text_only + [
            Message(
                role="user",
                content=[
                    TextContent(text="and this?"),
                    ImageContent(image_url=ImageUrlSpec(url="data:image/png;base64,AA")),
                ],
            )
        ]
        self.assertFalse(self.engine._conversation_has_images(text_only))
        self.assertTrue(self.engine._conversation_has_images(with_image))


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class VllmInitFailureTests(unittest.TestCase):
    def test_all_enabled_load_failures_include_model_error(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                backend="vllm",
                models={"broken-model": _make_settings(enabled=True)},
            ),
        )

        with mock.patch.object(
            VllmEngine,
            "_build_runtime",
            side_effect=RuntimeError("cuda out of memory"),
        ), mock.patch("app.engine.vllm.LOGGER.exception"):
            with self.assertRaisesRegex(
                ValueError,
                "no vLLM models could be loaded: broken-model: cuda out of memory",
            ):
                VllmEngine(settings)

    def test_no_enabled_models_keeps_existing_message(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                backend="vllm",
                models={"disabled-model": _make_settings(enabled=False)},
            ),
        )

        with self.assertRaisesRegex(ValueError, "^no enabled models could be loaded$"):
            VllmEngine(settings)


if __name__ == "__main__":
    unittest.main()
