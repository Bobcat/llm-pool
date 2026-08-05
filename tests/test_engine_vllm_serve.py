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
    import app.engine.vllm_serve as vllm_serve_module
    from app.schemas import AudioContent
    from app.schemas import AudioUrlSpec
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
class VllmServeEngineTests(unittest.TestCase):
    def test_chat_completion_payload_uses_default_top_k(self) -> None:
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults(top_k=11)
        request = ResponseRequest(model="gemma4", input="Hello")

        payload = engine._chat_completions_payload(
            runtime=mock.Mock(remote_model="gemma-local"),
            request=request,
            decoding=engine._resolve_decoding(request.decoding),
        )

        self.assertEqual(payload["top_k"], 11)

    def test_starts_vllm_serve_and_posts_multimodal_chat_completion(self) -> None:
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
                        model_path=None,
                        backend="vllm_serve",
                        vllm_model="/models/nvidia/Gemma-4-26B-A4B-NVFP4",
                        vllm_dtype="auto",
                        vllm_gpu_memory_utilization=0.55,
                        vllm_kv_cache_memory_bytes=2147483648,
                        vllm_kv_cache_dtype="fp8",
                        vllm_max_model_len=8192,
                        vllm_tensor_parallel_size=1,
                        vllm_trust_remote_code=True,
                        vllm_enforce_eager=True,
                        vllm_limit_mm_per_prompt=(("image", 1), ("audio", 1)),
                        vllm_mm_processor_kwargs=(("max_soft_tokens", 560),),
                        vllm_speculative_method="mtp",
                        vllm_speculative_model="google/gemma-4-26B-A4B-it-assistant",
                        vllm_speculative_moe_backend="triton",
                        vllm_speculative_attention_backend="triton_attn",
                        vllm_num_speculative_tokens=4,
                        vllm_serve_binary="/opt/vllm/bin/vllm",
                        vllm_serve_host="127.0.0.1",
                        vllm_serve_port=18090,
                        vllm_serve_model_alias="gemma-local",
                        vllm_serve_timeout_s=12.5,
                        vllm_serve_start_timeout_s=1.0,
                        vllm_serve_stop_timeout_s=2.0,
                        vllm_serve_library_path=("/cuda/lib",),
                        vllm_serve_env=(("VLLM_USE_FLASHINFER_SAMPLER", "0"),),
                        vllm_serve_api_key="local-secret",
                        vllm_serve_extra_args=(
                            "--tool-call-parser",
                            "gemma4",
                            "--reasoning-parser",
                            "gemma4",
                            "--enable-auto-tool-choice",
                        ),
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
            if request.full_url == "http://127.0.0.1:18090/v1/models":
                captured["health_timeout"] = timeout
                return FakeResponse({"data": [{"id": "gemma-local"}]})
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
            mock.patch.object(vllm_serve_module.subprocess, "Popen", side_effect=fake_popen),
            mock.patch.object(vllm_serve_module, "urlopen", side_effect=fake_urlopen),
        ):
            engine = vllm_serve_module.VllmServeEngine(settings)
            result = engine.complete(
                ResponseRequest(
                    model="gemma4",
                    input=[
                        TextContent(text="Describe this."),
                        ImageContent(image_url=ImageUrlSpec(url="data:image/png;base64,abc")),
                        AudioContent(audio_url=AudioUrlSpec(url="data:audio/wav;base64,abc")),
                    ],
                    instructions="Be terse.",
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "result",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    decoding=DecodingParams(
                        top_k=7,
                        top_p=0.8,
                        temperature=0.2,
                        max_tokens=9,
                        stop=["DONE"],
                    ),
                )
            )
            engine._models["gemma4"].close()

        command = captured["command"]
        self.assertEqual(command[0], "/opt/vllm/bin/vllm")
        self.assertEqual(command[1], "serve")
        self.assertEqual(command[2], "/models/nvidia/Gemma-4-26B-A4B-NVFP4")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18090")
        self.assertEqual(command[command.index("--served-model-name") + 1], "gemma-local")
        self.assertEqual(command[command.index("--dtype") + 1], "auto")
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.55")
        self.assertEqual(command[command.index("--kv-cache-memory-bytes") + 1], "2147483648")
        self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "fp8")
        self.assertEqual(command[command.index("--max-model-len") + 1], "8192")
        self.assertIn("--trust-remote-code", command)
        self.assertIn("--enforce-eager", command)
        self.assertEqual(
            json.loads(command[command.index("--limit-mm-per-prompt") + 1]),
            {"image": 1, "audio": 1},
        )
        self.assertEqual(
            json.loads(command[command.index("--mm-processor-kwargs") + 1]),
            {"max_soft_tokens": 560},
        )
        self.assertEqual(
            json.loads(command[command.index("--speculative-config") + 1]),
            {
                "method": "mtp",
                "num_speculative_tokens": 4,
                "model": "google/gemma-4-26B-A4B-it-assistant",
                "moe_backend": "triton",
                "attention_backend": "triton_attn",
            },
        )
        self.assertEqual(command[command.index("--api-key") + 1], "local-secret")
        self.assertIn("--tool-call-parser", command)
        self.assertIn("--reasoning-parser", command)
        self.assertIn("--enable-auto-tool-choice", command)
        popen_env = captured["popen_kwargs"]["env"]
        self.assertTrue(popen_env["PATH"].startswith("/opt/vllm/bin"))
        self.assertTrue(popen_env["LD_LIBRARY_PATH"].startswith("/cuda/lib"))
        self.assertEqual(popen_env["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertEqual(captured["chat_url"], "http://127.0.0.1:18090/v1/chat/completions")
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
                            {
                                "type": "audio_url",
                                "audio_url": {
                                    "url": "data:audio/wav;base64,abc",
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.2,
                "top_k": 7,
                "top_p": 0.8,
                "max_tokens": 9,
                "stop": ["DONE"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
        self.assertEqual(result.text, "looks good")
        self.assertEqual(result.metrics.engine_prompt_tokens, 11)
        self.assertEqual(result.metrics.engine_output_tokens, 3)
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)


if __name__ == "__main__":
    unittest.main()
