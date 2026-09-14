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
    import app.engine.sglang_serve as sglang_serve_module
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
    pid = 4321

    def __init__(self) -> None:
        self.return_code: int | None = None

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.return_code = 0
        return 0


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class SglangServeEngineTests(unittest.TestCase):
    def test_omits_speculative_flags_when_algorithm_is_disabled(self) -> None:
        settings = ModelSettings(
            model_path=None,
            backend="sglang_serve",
            sglang_model="/models/gemma4",
            sglang_speculative_algorithm=None,
            sglang_speculative_draft_model="assistant",
        )

        command = sglang_serve_module.SglangServeEngine.__new__(
            sglang_serve_module.SglangServeEngine
        )._command(
            settings=settings,
            model_ref="/models/gemma4",
            host="127.0.0.1",
            port=18092,
            remote_model="gemma4",
        )

        self.assertNotIn("--speculative-algorithm", command)
        self.assertNotIn("--speculative-draft-model-path", command)

    def test_starts_sglang_and_posts_multimodal_chat_completion(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                decoding=DecodingDefaults(max_tokens=32),
                models={
                    "gemma4": ModelSettings(
                        model_path=None,
                        backend="sglang_serve",
                        target_inflight=4,
                        enable_thinking=False,
                        sglang_model="/models/nvidia/Gemma-4-26B-A4B-NVFP4",
                        sglang_context_length=20480,
                        sglang_mem_fraction_static=0.3,
                        sglang_max_total_tokens=20480,
                        sglang_chunked_prefill_size=8192,
                        sglang_kv_cache_dtype="fp8_e4m3",
                        sglang_quantization="modelopt_fp4",
                        sglang_tensor_parallel_size=1,
                        sglang_trust_remote_code=True,
                        sglang_attention_backend="triton",
                        sglang_fp4_gemm_backend="auto",
                        sglang_speculative_algorithm="NEXTN",
                        sglang_speculative_draft_model=(
                            "google/gemma-4-26B-A4B-it-assistant"
                        ),
                        sglang_speculative_num_steps=5,
                        sglang_speculative_num_draft_tokens=6,
                        sglang_speculative_eagle_topk=1,
                        sglang_serve_binary="/opt/sglang/bin/sglang",
                        sglang_serve_host="127.0.0.1",
                        sglang_serve_port=18092,
                        sglang_serve_model_alias="gemma-local",
                        sglang_serve_timeout_s=12.5,
                        sglang_serve_start_timeout_s=1.0,
                        sglang_serve_stop_timeout_s=2.0,
                        sglang_serve_library_path=("/cuda/lib",),
                        sglang_serve_env=(
                            ("CUDA_HOME", "/cuda"),
                            ("MAX_JOBS", "4"),
                        ),
                        sglang_serve_api_key="local-secret",
                        sglang_serve_reasoning_parser="gemma4",
                        sglang_serve_tool_parser="gemma4",
                        sglang_serve_extra_args=(
                            "--cuda-graph-backend-decode",
                            "disabled",
                        ),
                    )
                },
            )
        )
        process = FakeProcess()
        captured: dict[str, object] = {}

        def fake_popen(command, **kwargs):
            captured["command"] = list(command)
            captured["popen_kwargs"] = kwargs
            return process

        def fake_urlopen(request, *, timeout):
            if request.full_url == "http://127.0.0.1:18092/v1/models":
                captured["health_timeout"] = timeout
                return FakeResponse({"data": [{"id": "gemma-local"}]})
            captured["chat_url"] = request.full_url
            captured["chat_headers"] = dict(request.header_items())
            captured["chat_timeout"] = timeout
            captured["chat_body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "  looks good  "}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                }
            )

        with (
            mock.patch.object(
                sglang_serve_module.subprocess,
                "Popen",
                side_effect=fake_popen,
            ),
            mock.patch.object(sglang_serve_module, "urlopen", side_effect=fake_urlopen),
            mock.patch.object(sglang_serve_module.os, "killpg") as killpg,
        ):
            engine = sglang_serve_module.SglangServeEngine(settings)
            result = engine.complete(
                ResponseRequest(
                    model="gemma4",
                    input=[
                        TextContent(text="Describe this."),
                        ImageContent(
                            image_url=ImageUrlSpec(url="data:image/png;base64,abc")
                        ),
                    ],
                    instructions="Be terse.",
                    thinking="disabled",
                    decoding=DecodingParams(
                        top_k=7,
                        top_p=0.8,
                        temperature=0.2,
                        repetition_penalty=1.1,
                        max_tokens=9,
                        stop=["DONE"],
                    ),
                )
            )
            engine._models["gemma4"].close()

        command = captured["command"]
        self.assertEqual(command[:3], ["/opt/sglang/bin/sglang", "serve", "--model-path"])
        self.assertEqual(command[3], "/models/nvidia/Gemma-4-26B-A4B-NVFP4")
        expected_arguments = {
            "--host": "127.0.0.1",
            "--port": "18092",
            "--served-model-name": "gemma-local",
            "--tp-size": "1",
            "--context-length": "20480",
            "--mem-fraction-static": "0.3",
            "--max-running-requests": "4",
            "--max-total-tokens": "20480",
            "--chunked-prefill-size": "8192",
            "--kv-cache-dtype": "fp8_e4m3",
            "--quantization": "modelopt_fp4",
            "--attention-backend": "triton",
            "--fp4-gemm-backend": "auto",
            "--reasoning-parser": "gemma4",
            "--tool-call-parser": "gemma4",
            "--speculative-algorithm": "NEXTN",
            "--speculative-draft-model-path": (
                "google/gemma-4-26B-A4B-it-assistant"
            ),
            "--speculative-num-steps": "5",
            "--speculative-num-draft-tokens": "6",
            "--speculative-eagle-topk": "1",
            "--api-key": "local-secret",
            "--cuda-graph-backend-decode": "disabled",
        }
        for argument, expected_value in expected_arguments.items():
            self.assertEqual(command[command.index(argument) + 1], expected_value)
        self.assertIn("--trust-remote-code", command)

        popen_kwargs = captured["popen_kwargs"]
        self.assertTrue(popen_kwargs["start_new_session"])
        self.assertTrue(popen_kwargs["env"]["PATH"].startswith("/opt/sglang/bin:/cuda/bin"))
        self.assertTrue(popen_kwargs["env"]["LD_LIBRARY_PATH"].startswith("/cuda/lib"))
        self.assertEqual(popen_kwargs["env"]["MAX_JOBS"], "4")
        self.assertEqual(captured["health_timeout"], 1.0)
        self.assertEqual(captured["chat_url"], "http://127.0.0.1:18092/v1/chat/completions")
        self.assertEqual(captured["chat_timeout"], 12.5)
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
                "top_k": 7,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
                "max_tokens": 9,
                "stop": ["DONE"],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(result.text, "looks good")
        self.assertEqual(result.metrics.engine_prompt_tokens, 11)
        self.assertEqual(result.metrics.engine_output_tokens, 3)
        killpg.assert_called_once_with(4321, sglang_serve_module.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
