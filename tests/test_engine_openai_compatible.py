from __future__ import annotations

import importlib.util
import json
import os
import unittest
from unittest import mock
from urllib.error import HTTPError

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import AppSettings
    from app.config import DecodingDefaults
    from app.config import EngineSettings
    from app.config import ModelSettings
    from app.engine.common import ResolvedDecoding
    import app.engine.openai_compatible as openai_compatible_module
    from app.schemas import DecodingParams
    from app.schemas import ResponseRequest


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def close(self) -> None:
        pass


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class OpenAICompatibleEngineTests(unittest.TestCase):
    def test_complete_posts_chat_completion_and_extracts_usage(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                decoding=DecodingDefaults(
                    top_p=1.0,
                    temperature=0.1,
                    max_tokens=32,
                    stop=[],
                ),
                models={
                    "remote-model": ModelSettings(
                        model_path=None,
                        backend="openai_compatible",
                        remote_api_kind="chat_completions",
                        remote_base_url="https://api.example.com/v1/",
                        remote_api_key_env="EXAMPLE_API_KEY",
                        remote_model="provider-model",
                        remote_timeout_s=12.5,
                        remote_thinking="disabled",
                    ),
                },
            ),
        )
        captured: dict[str, object] = {}

        def fake_urlopen(request, *, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "  done  ",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                }
            )

        previous = os.environ.get("EXAMPLE_API_KEY")
        os.environ["EXAMPLE_API_KEY"] = "secret"
        try:
            with mock.patch.object(openai_compatible_module, "urlopen", side_effect=fake_urlopen):
                engine = openai_compatible_module.OpenAICompatibleEngine(settings)
                result = engine.complete(
                    ResponseRequest(
                        model="remote-model",
                        input="Hello",
                        instructions="Be brief.",
                        allow_remote=True,
                        decoding=DecodingParams(
                            beam_size=2,
                            top_k=7,
                            top_p=0.8,
                            temperature=0.2,
                            repetition_penalty=1.2,
                            max_tokens=9,
                            stop=["DONE"],
                        ),
                    )
                )
        finally:
            if previous is None:
                os.environ.pop("EXAMPLE_API_KEY", None)
            else:
                os.environ["EXAMPLE_API_KEY"] = previous

        self.assertEqual(result.text, "done")
        self.assertEqual(result.metrics.engine_prompt_tokens, 12)
        self.assertEqual(result.metrics.engine_output_tokens, 5)
        self.assertIsNotNone(result.metrics.engine_tokens_per_second)
        self.assertEqual(captured["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(captured["timeout"], 12.5)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(
            captured["body"],
            {
                "model": "provider-model",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hello"},
                ],
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 9,
                "stop": ["DONE"],
                "thinking": {"type": "disabled"},
            },
        )

    def test_remote_thinking_request_override_takes_precedence(self) -> None:
        engine = openai_compatible_module.OpenAICompatibleEngine.__new__(
            openai_compatible_module.OpenAICompatibleEngine
        )
        runtime = openai_compatible_module.OpenAICompatibleModelRuntime(
            config=ModelSettings(
                model_path=None,
                backend="openai_compatible",
                remote_thinking="disabled",
            ),
            base_url="https://api.example.com/v1",
            api_key_env="EXAMPLE_API_KEY",
            remote_model="provider-model",
            timeout_s=1.0,
        )

        payload = engine._chat_completions_payload(
            runtime=runtime,
            request=ResponseRequest(
                model="remote-model",
                input="Hello",
                thinking="enabled",
            ),
            decoding=ResolvedDecoding(
                beam_size=1,
                top_k=1,
                top_p=0.8,
                temperature=0.2,
                repetition_penalty=1.0,
                max_tokens=16,
                stop=[],
            ),
        )

        self.assertEqual(payload["thinking"], {"type": "enabled"})

    def test_kimi_thinking_omits_non_one_temperature(self) -> None:
        engine = openai_compatible_module.OpenAICompatibleEngine.__new__(
            openai_compatible_module.OpenAICompatibleEngine
        )
        runtime = openai_compatible_module.OpenAICompatibleModelRuntime(
            config=ModelSettings(
                model_path=None,
                backend="openai_compatible",
                remote_thinking="disabled",
            ),
            base_url="https://api.moonshot.ai/v1",
            api_key_env="EXAMPLE_API_KEY",
            remote_model="kimi-k2.6",
            timeout_s=1.0,
        )

        payload = engine._chat_completions_payload(
            runtime=runtime,
            request=ResponseRequest(
                model="kimi-k2.6",
                input="Hello",
                thinking="enabled",
            ),
            decoding=ResolvedDecoding(
                beam_size=1,
                top_k=1,
                top_p=0.95,
                temperature=0.6,
                repetition_penalty=1.0,
                max_tokens=16,
                stop=[],
            ),
        )

        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["thinking"], {"type": "enabled"})

    def test_kimi_disabled_thinking_omits_unsupported_sampling_values(self) -> None:
        engine = openai_compatible_module.OpenAICompatibleEngine.__new__(
            openai_compatible_module.OpenAICompatibleEngine
        )
        runtime = openai_compatible_module.OpenAICompatibleModelRuntime(
            config=ModelSettings(
                model_path=None,
                backend="openai_compatible",
                remote_thinking="disabled",
            ),
            base_url="https://api.moonshot.ai/v1",
            api_key_env="EXAMPLE_API_KEY",
            remote_model="kimi-k2.6",
            timeout_s=1.0,
        )

        payload = engine._chat_completions_payload(
            runtime=runtime,
            request=ResponseRequest(
                model="kimi-k2.6",
                input="Hello",
                thinking="disabled",
            ),
            decoding=ResolvedDecoding(
                beam_size=1,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                repetition_penalty=1.0,
                max_tokens=16,
                stop=[],
            ),
        )

        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_build_runtime_requires_api_key_environment_variable(self) -> None:
        engine = openai_compatible_module.OpenAICompatibleEngine.__new__(
            openai_compatible_module.OpenAICompatibleEngine
        )
        settings = ModelSettings(
            model_path=None,
            backend="openai_compatible",
            remote_api_kind="chat_completions",
            remote_base_url="https://api.example.com/v1",
            remote_api_key_env="MISSING_API_KEY",
            remote_model="provider-model",
        )

        previous = os.environ.get("MISSING_API_KEY")
        os.environ.pop("MISSING_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError) as exc_info:
                engine._build_runtime(settings)
        finally:
            if previous is not None:
                os.environ["MISSING_API_KEY"] = previous

        self.assertEqual(
            str(exc_info.exception),
            "missing API key environment variable: MISSING_API_KEY",
        )

    def test_http_error_message_includes_upstream_error_body(self) -> None:
        exc = HTTPError(
            url="https://api.example.com/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=FakeResponse(
                {
                    "error": {
                        "message": "Invalid request: unsupported image format",
                        "type": "invalid_request_error",
                    }
                }
            ),
        )

        mapped = openai_compatible_module.OpenAICompatibleEngine._map_http_error(
            openai_compatible_module.OpenAICompatibleEngine.__new__(
                openai_compatible_module.OpenAICompatibleEngine
            ),
            exc,
        )

        self.assertEqual(mapped.code, "upstream_invalid_request")
        self.assertEqual(mapped.status_code, 502)
        self.assertIn(
            "upstream chat completion rejected the request with HTTP 400: "
            "Invalid request: unsupported image format",
            mapped.message,
        )


if __name__ == "__main__":
    unittest.main()
