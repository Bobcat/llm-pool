from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import tempfile
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
    def test_rejects_max_running_requests_in_extra_args(self) -> None:
        for extra_args in (
            ("--max-running-requests", "8"),
            ("--max-running-requests=8",),
        ):
            with self.subTest(extra_args=extra_args):
                settings = ModelSettings(
                    model_path=None,
                    backend="sglang_serve",
                    sglang_model="/models/gemma4",
                    sglang_serve_extra_args=extra_args,
                )

                with self.assertRaisesRegex(ValueError, "controlled by target_inflight"):
                    sglang_serve_module.SglangServeEngine.__new__(
                        sglang_serve_module.SglangServeEngine
                    )._command(
                        settings=settings,
                        model_ref="/models/gemma4",
                        host="127.0.0.1",
                        port=18092,
                        remote_model="gemma4",
                    )

    def test_derives_topk_one_draft_token_count(self) -> None:
        settings = ModelSettings(
            model_path=None,
            backend="sglang_serve",
            sglang_model="/models/gemma4",
            sglang_speculative_algorithm="NEXTN",
            sglang_speculative_num_steps=5,
            sglang_speculative_num_draft_tokens=4,
            sglang_speculative_eagle_topk=1,
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

        self.assertEqual(
            command[command.index("--speculative-num-draft-tokens") + 1],
            "6",
        )

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

    def test_build_runtime_closes_process_after_readiness_failure(self) -> None:
        settings = ModelSettings(
            model_path=None,
            backend="sglang_serve",
            sglang_model="/models/gemma4",
            sglang_serve_port=18092,
        )
        output_log = tempfile.TemporaryFile()
        self.addCleanup(output_log.close)
        engine = sglang_serve_module.SglangServeEngine.__new__(
            sglang_serve_module.SglangServeEngine
        )

        with (
            mock.patch.object(
                engine,
                "_start_process",
                return_value=(FakeProcess(), output_log),
            ),
            mock.patch.object(
                engine,
                "_wait_until_ready",
                side_effect=RuntimeError("not ready"),
            ),
            mock.patch.object(
                sglang_serve_module.SglangServeModelRuntime,
                "close",
            ) as close,
            self.assertRaisesRegex(RuntimeError, "not ready"),
        ):
            engine._build_runtime("gemma4", settings)

        close.assert_called_once_with()

    def test_close_kills_process_group_after_timeout(self) -> None:
        process = FakeProcess()
        process.wait = mock.Mock(
            side_effect=[
                subprocess.TimeoutExpired(cmd="sglang", timeout=2.0),
                -9,
            ]
        )
        runtime = self._runtime(process, stop_timeout_s=2.0)

        with mock.patch.object(sglang_serve_module.os, "killpg") as killpg:
            runtime.close()

        self.assertEqual(
            killpg.call_args_list,
            [mock.call(4321, signal.SIGTERM), mock.call(4321, signal.SIGKILL)],
        )
        self.assertTrue(runtime.output_log.closed)

    def test_close_escalates_after_process_leader_exits(self) -> None:
        process = FakeProcess()
        process.return_code = 1
        process.wait = mock.Mock()
        runtime = self._runtime(process, stop_timeout_s=0.0)

        with mock.patch.object(sglang_serve_module.os, "killpg") as killpg:
            runtime.close()
            runtime.close()

        self.assertEqual(
            killpg.call_args_list,
            [mock.call(4321, signal.SIGTERM), mock.call(4321, signal.SIGKILL)],
        )
        process.wait.assert_not_called()
        self.assertTrue(runtime.output_log.closed)

    def test_close_stops_when_orphan_group_is_gone_or_not_permitted(self) -> None:
        for terminal_error in (ProcessLookupError, PermissionError):
            with self.subTest(terminal_error=terminal_error):
                process = FakeProcess()
                process.return_code = 1
                process.wait = mock.Mock()
                runtime = self._runtime(process, stop_timeout_s=2.0)

                with (
                    mock.patch.object(
                        sglang_serve_module.os,
                        "killpg",
                        side_effect=[None, terminal_error],
                    ) as killpg,
                    mock.patch.object(
                        sglang_serve_module.time,
                        "monotonic",
                        return_value=10.0,
                    ),
                ):
                    runtime.close()

                self.assertEqual(
                    killpg.call_args_list,
                    [mock.call(4321, signal.SIGTERM), mock.call(4321, 0)],
                )
                process.wait.assert_not_called()
                self.assertTrue(runtime.output_log.closed)

    def test_close_logs_when_process_survives_sigkill(self) -> None:
        process = FakeProcess()
        process.wait = mock.Mock(
            side_effect=[
                subprocess.TimeoutExpired(cmd="sglang", timeout=2.0),
                subprocess.TimeoutExpired(cmd="sglang", timeout=5.0),
            ]
        )
        runtime = self._runtime(process, stop_timeout_s=2.0)

        with (
            mock.patch.object(sglang_serve_module.os, "killpg") as killpg,
            self.assertLogs(sglang_serve_module.LOGGER, level="WARNING") as logs,
        ):
            runtime.close()

        self.assertEqual(
            killpg.call_args_list,
            [mock.call(4321, signal.SIGTERM), mock.call(4321, signal.SIGKILL)],
        )
        self.assertIn("did not exit after SIGKILL", logs.output[0])
        self.assertTrue(runtime.output_log.closed)

    def test_startup_exit_includes_process_output_tail(self) -> None:
        process = FakeProcess()
        process.return_code = 17
        output_log = tempfile.TemporaryFile()
        output_log.write(b"engine initialized\naddress already in use\n")
        output_log.flush()
        runtime = self._runtime(process, output_log=output_log)
        engine = sglang_serve_module.SglangServeEngine.__new__(
            sglang_serve_module.SglangServeEngine
        )

        with self.assertRaises(RuntimeError) as exc_info:
            engine._wait_until_ready(runtime, 1.0)

        self.assertIn("exited during startup with code 17", str(exc_info.exception))
        self.assertIn("address already in use", str(exc_info.exception))
        output_log.close()

    def test_empty_final_content_with_reasoning_is_incomplete(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "unfinished reasoning",
                    },
                }
            ]
        }

        with self.assertRaises(
            sglang_serve_module.BackendExecutionError
        ) as exc_info:
            sglang_serve_module.SglangServeEngine._extract_text(payload)

        self.assertEqual(
            exc_info.exception.code,
            "sglang_serve_incomplete_response",
        )

    @staticmethod
    def _runtime(
        process: FakeProcess,
        *,
        stop_timeout_s: float = 2.0,
        output_log=None,
    ):
        return sglang_serve_module.SglangServeModelRuntime(
            config=mock.Mock(),
            process=process,
            base_url="http://127.0.0.1:18092/v1",
            health_url="http://127.0.0.1:18092/v1/models",
            remote_model="gemma-local",
            timeout_s=12.5,
            api_key=None,
            stop_timeout_s=stop_timeout_s,
            output_log=output_log or tempfile.TemporaryFile(),
        )


if __name__ == "__main__":
    unittest.main()
