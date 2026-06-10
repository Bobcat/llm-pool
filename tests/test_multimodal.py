"""Tests for the Phase 1 multimodal-input extension.

Covers:
- Polymorphic ResponseRequest.input (string vs list[ContentItem])
- text_input_or_raise / has_image_content helpers
- Backwards-compat of string-input path
- Config parsing for modalities
- OpenAI-compatible payload shape for both string and list input
- Stub backend rejection of image content (MODALITY_UNSUPPORTED)
- AdminModelEntry capabilities field
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import AppSettings
    from app.config import DecodingDefaults
    from app.config import EngineSettings
    from app.config import ModelSettings
    from app.config import load_settings
    from app.engine import BackendExecutionError
    import app.engine.openai_compatible as openai_compatible_module
    from app.engine.stub import StubEngine
    from app.schemas import ImageContent
    from app.schemas import ImageUrlSpec
    from app.schemas import ModalityUnsupportedError
    from app.schemas import ResponseRequest
    from app.schemas import TextContent


class _FakeUpstreamResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class ResponseRequestSchemaTests(unittest.TestCase):
    def test_string_input_is_accepted_and_preserved(self) -> None:
        request = ResponseRequest(model="m", input="Hello world")
        self.assertEqual(request.input, "Hello world")
        self.assertFalse(request.has_image_content)
        self.assertEqual(request.text_input_or_raise(), "Hello world")

    def test_text_only_content_list_joins_to_plain_text(self) -> None:
        request = ResponseRequest(
            model="m",
            input=[
                TextContent(text="Hello "),
                TextContent(text="world"),
            ],
        )
        self.assertFalse(request.has_image_content)
        self.assertEqual(request.text_input_or_raise(), "Hello world")

    def test_image_content_marks_has_image_content_and_raises_on_text_only(self) -> None:
        request = ResponseRequest(
            model="m",
            input=[
                TextContent(text="Describe this image:"),
                ImageContent(image_url=ImageUrlSpec(url="https://example.com/image.png")),
            ],
        )
        self.assertTrue(request.has_image_content)
        with self.assertRaises(ModalityUnsupportedError):
            request.text_input_or_raise()

    def test_content_item_discriminator_rejects_unknown_type(self) -> None:
        with self.assertRaises(Exception):  # pydantic ValidationError
            ResponseRequest.model_validate(
                {"model": "m", "input": [{"type": "audio", "audio": "..."}]}
            )

    def test_image_url_parses_from_json_payload(self) -> None:
        request = ResponseRequest.model_validate(
            {
                "model": "m",
                "input": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0..."},
                    },
                ],
            }
        )
        self.assertTrue(request.has_image_content)
        # default detail should be "auto"
        image_item = request.input[1]
        self.assertIsInstance(image_item, ImageContent)
        self.assertEqual(image_item.image_url.detail, "auto")


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class ConfigModalitiesTests(unittest.TestCase):
    def test_defaults_to_text_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "backend": "stub",
                            "models": {
                                "m": {
                                    "model_path": "/tmp/m",
                                    "enabled": True,
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            settings = load_settings(settings_path)
        self.assertEqual(settings.engine.models["m"].modalities, ("text",))

    def test_explicit_image_modality_keeps_text_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "backend": "stub",
                            "models": {
                                "m": {
                                    "model_path": "/tmp/m",
                                    "enabled": True,
                                    "modalities": ["image"],
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            settings = load_settings(settings_path)
        self.assertEqual(settings.engine.models["m"].modalities, ("text", "image"))

    def test_both_modalities_preserved_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "backend": "stub",
                            "models": {
                                "m": {
                                    "model_path": "/tmp/m",
                                    "enabled": True,
                                    "modalities": ["text", "image"],
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            settings = load_settings(settings_path)
        self.assertEqual(settings.engine.models["m"].modalities, ("text", "image"))

    def test_invalid_modality_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "backend": "stub",
                            "models": {
                                "m": {
                                    "model_path": "/tmp/m",
                                    "enabled": True,
                                    "modalities": ["audio"],
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_settings(settings_path)


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class OpenAICompatibleMultimodalPayloadTests(unittest.TestCase):
    def _build_engine_and_capture(self, request: ResponseRequest):
        settings = AppSettings(
            engine=EngineSettings(
                decoding=DecodingDefaults(top_p=1.0, temperature=0.1, max_tokens=32, stop=[]),
                models={
                    "remote-model": ModelSettings(
                        model_path=None,
                        backend="openai_compatible",
                        remote_api_kind="chat_completions",
                        remote_base_url="https://api.example.com/v1/",
                        remote_api_key_env="EXAMPLE_API_KEY",
                        remote_model="provider-model",
                        remote_timeout_s=12.5,
                    ),
                },
            ),
        )
        captured: dict[str, object] = {}

        def fake_urlopen(req, *, timeout):
            del timeout
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeUpstreamResponse(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

        previous = os.environ.get("EXAMPLE_API_KEY")
        os.environ["EXAMPLE_API_KEY"] = "secret"
        try:
            with mock.patch.object(openai_compatible_module, "urlopen", side_effect=fake_urlopen):
                engine = openai_compatible_module.OpenAICompatibleEngine(settings)
                engine.complete(request)
        finally:
            if previous is None:
                os.environ.pop("EXAMPLE_API_KEY", None)
            else:
                os.environ["EXAMPLE_API_KEY"] = previous
        return captured

    def test_string_input_passes_through_as_string_content(self) -> None:
        request = ResponseRequest(model="remote-model", input="Hello")
        captured = self._build_engine_and_capture(request)
        user_message = captured["body"]["messages"][1]
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(user_message["content"], "Hello")

    def test_multimodal_input_passes_through_as_content_list(self) -> None:
        request = ResponseRequest(
            model="remote-model",
            input=[
                TextContent(text="Describe this image:"),
                ImageContent(image_url=ImageUrlSpec(url="https://example.com/image.png")),
            ],
        )
        captured = self._build_engine_and_capture(request)
        user_message = captured["body"]["messages"][1]
        self.assertEqual(user_message["role"], "user")
        self.assertIsInstance(user_message["content"], list)
        self.assertEqual(len(user_message["content"]), 2)
        self.assertEqual(user_message["content"][0]["type"], "text")
        self.assertEqual(user_message["content"][0]["text"], "Describe this image:")
        self.assertEqual(user_message["content"][1]["type"], "image_url")
        self.assertEqual(
            user_message["content"][1]["image_url"]["url"],
            "https://example.com/image.png",
        )


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class StubBackendMultimodalGuardTests(unittest.TestCase):
    def _build_stub(self, modalities: tuple[str, ...] = ("text",)) -> StubEngine:
        settings = AppSettings(
            engine=EngineSettings(
                backend="stub",
                models={
                    "test-model": ModelSettings(
                        model_path="/tmp/test-model",
                        enabled=True,
                        modalities=modalities,
                    ),
                },
            ),
        )
        return StubEngine(settings)

    def test_string_input_works_unchanged(self) -> None:
        engine = self._build_stub()
        result = engine.complete(ResponseRequest(model="test-model", input="Hello"))
        self.assertEqual(result.text, "Hello")

    def test_text_only_content_list_works(self) -> None:
        engine = self._build_stub()
        result = engine.complete(
            ResponseRequest(
                model="test-model",
                input=[TextContent(text="Hello "), TextContent(text="world")],
            )
        )
        self.assertEqual(result.text, "Hello world")

    def test_image_content_raises_modality_unsupported(self) -> None:
        engine = self._build_stub()
        request = ResponseRequest(
            model="test-model",
            input=[
                TextContent(text="Describe this:"),
                ImageContent(image_url=ImageUrlSpec(url="https://example.com/image.png")),
            ],
        )
        with self.assertRaises(BackendExecutionError) as exc_info:
            engine.complete(request)
        self.assertEqual(exc_info.exception.code, "modality_unsupported")
        self.assertEqual(exc_info.exception.status_code, 400)


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class AdminCapabilitiesExposureTests(unittest.TestCase):
    def test_stub_admin_payload_reports_default_text_capability(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                backend="stub",
                models={
                    "m": ModelSettings(
                        model_path="/tmp/m",
                        enabled=True,
                    ),
                },
            ),
        )
        engine = StubEngine(settings)
        payload = engine.admin_models_payload(settings)
        entry = payload["models"][0]
        self.assertEqual(
            entry["capabilities"], {"modalities": ["text"], "multi_turn": False}
        )

    def test_stub_admin_payload_reports_image_capability_when_configured(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                backend="stub",
                models={
                    "m": ModelSettings(
                        model_path="/tmp/m",
                        enabled=True,
                        modalities=("text", "image"),
                    ),
                },
            ),
        )
        engine = StubEngine(settings)
        payload = engine.admin_models_payload(settings)
        entry = payload["models"][0]
        self.assertEqual(
            entry["capabilities"],
            {"modalities": ["text", "image"], "multi_turn": False},
        )


if __name__ == "__main__":
    unittest.main()
