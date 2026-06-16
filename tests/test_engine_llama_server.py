from __future__ import annotations

import importlib.util
import json
import unittest
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import AppSettings
    from app.config import DecodingDefaults
    from app.config import EngineSettings
    from app.config import ModelSettings
    import app.engine.llama_server as llama_server_module
    from app.schemas import DecodingParams
    from app.schemas import ImageContent
    from app.schemas import ImageUrlSpec
    from app.schemas import ResponseRequest
    from app.schemas import TextContent


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class FakeProcess:
    def __init__(self) -> None:
        self.return_code: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = 0

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.return_code is None:
            self.return_code = 0
        return self.return_code


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class LlamaServerEngineTests(unittest.TestCase):
    def test_starts_llama_server_and_posts_multimodal_chat_completion(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                decoding=DecodingDefaults(
                    top_p=1.0,
                    temperature=0.1,
                    max_tokens=32,
                    stop=[],
                ),
                models={
                    "gemma4": ModelSettings(
                        model_path="/models/gemma.gguf",
                        backend="llama_server",
                        llama_server_binary="/opt/llama-server",
                        llama_server_host="127.0.0.1",
                        llama_server_port=18089,
                        llama_server_model_alias="gemma-local",
                        llama_server_timeout_s=12.5,
                        llama_server_start_timeout_s=1.0,
                        llama_server_stop_timeout_s=2.0,
                        llama_server_library_path=("/cuda/lib",),
                        llama_server_api_key="local-secret",
                        llama_server_n_ctx=4096,
                        llama_server_n_gpu_layers="999",
                        llama_server_flash_attn="on",
                        llama_server_mmproj_path="/models/mmproj.gguf",
                        llama_server_image_max_tokens=512,
                        llama_server_draft_model_path="/models/mtp.gguf",
                        llama_server_spec_type="draft-mtp",
                        llama_server_spec_draft_n_max=4,
                        llama_server_spec_draft_p_min=0.25,
                        llama_server_spec_draft_ngl="999",
                        llama_server_reasoning="off",
                        llama_server_extra_args=("--jinja",),
                    ),
                },
            ),
        )
        process = FakeProcess()
        captured: dict[str, object] = {}

        def fake_popen(command, **kwargs):
            captured["command"] = list(command)
            captured["popen_kwargs"] = kwargs
            return process

        def fake_urlopen(request, *, timeout):
            if request.full_url == "http://127.0.0.1:18089/health":
                captured["health_timeout"] = timeout
                return FakeResponse({"status": "ok"})
            captured["chat_url"] = request.full_url
            captured["chat_headers"] = dict(request.header_items())
            captured["chat_timeout"] = timeout
            captured["chat_body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "  looks good  "}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 3,
                    },
                }
            )

        with (
            mock.patch.object(llama_server_module.subprocess, "Popen", side_effect=fake_popen),
            mock.patch.object(llama_server_module, "urlopen", side_effect=fake_urlopen),
        ):
            engine = llama_server_module.LlamaServerEngine(settings)
            result = engine.complete(
                ResponseRequest(
                    model="gemma4",
                    input=[
                        TextContent(text="Describe this."),
                        ImageContent(image_url=ImageUrlSpec(url="data:image/png;base64,abc")),
                    ],
                    instructions="Be terse.",
                    decoding=DecodingParams(
                        top_p=0.8,
                        temperature=0.2,
                        max_tokens=9,
                        stop=["DONE"],
                    ),
                )
            )
            engine._models["gemma4"].close()

        command = captured["command"]
        self.assertEqual(command[0], "/opt/llama-server")
        self.assertEqual(command[command.index("--model") + 1], "/models/gemma.gguf")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18089")
        self.assertEqual(command[command.index("--alias") + 1], "gemma-local")
        self.assertIn("--no-ui", command)
        self.assertEqual(command[command.index("-fa") + 1], "on")
        self.assertEqual(command[command.index("-c") + 1], "4096")
        self.assertEqual(command[command.index("-ngl") + 1], "999")
        self.assertEqual(command[command.index("--mmproj") + 1], "/models/mmproj.gguf")
        self.assertEqual(command[command.index("--image-max-tokens") + 1], "512")
        self.assertEqual(command[command.index("--model-draft") + 1], "/models/mtp.gguf")
        self.assertEqual(command[command.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "4")
        self.assertEqual(command[command.index("--spec-draft-p-min") + 1], "0.25")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "999")
        self.assertEqual(command[command.index("--reasoning") + 1], "off")
        self.assertEqual(command[command.index("--api-key") + 1], "local-secret")
        self.assertIn("--jinja", command)
        popen_env = captured["popen_kwargs"]["env"]
        self.assertTrue(popen_env["LD_LIBRARY_PATH"].startswith("/cuda/lib"))
        self.assertEqual(captured["chat_url"], "http://127.0.0.1:18089/v1/chat/completions")
        self.assertEqual(captured["chat_timeout"], 12.5)
        self.assertEqual(captured["health_timeout"], 1.0)
        self.assertEqual(captured["chat_headers"]["Authorization"], "Bearer local-secret")
        self.assertEqual(
            captured["chat_body"],
            {
                "model": "gemma-local",
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,abc",
                                    "detail": "auto",
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 9,
                "stop": ["DONE"],
            },
        )
        self.assertEqual(result.text, "looks good")
        self.assertEqual(result.metrics.engine_prompt_tokens, 11)
        self.assertEqual(result.metrics.engine_output_tokens, 3)
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)


if __name__ == "__main__":
    unittest.main()
