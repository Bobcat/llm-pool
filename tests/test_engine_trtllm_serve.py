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
    import app.engine.trtllm_serve as trtllm_serve_module
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
        self.pid = 4242
        self.return_code: int | None = None

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.return_code = 0
        return self.return_code


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class TrtllmServeEngineTests(unittest.TestCase):
    def test_starts_server_and_posts_multimodal_chat_completion(self) -> None:
        settings = AppSettings(
            engine=EngineSettings(
                decoding=DecodingDefaults(
                    top_k=11,
                    top_p=1.0,
                    temperature=0.1,
                    repetition_penalty=1.0,
                    max_tokens=32,
                    stop=[],
                ),
                models={
                    "gemma4": ModelSettings(
                        model_path=None,
                        backend="trtllm_serve",
                        prompt_format="gemma4_template",
                        enable_thinking=True,
                        trtllm_model="/models/nvidia/Gemma-4-26B-A4B-NVFP4",
                        trtllm_trust_remote_code=True,
                        trtllm_serve_binary="/opt/trtllm/bin/trtllm-serve",
                        trtllm_serve_host="127.0.0.1",
                        trtllm_serve_port=18091,
                        trtllm_serve_model_alias="gemma-local",
                        trtllm_serve_timeout_s=12.5,
                        trtllm_serve_start_timeout_s=1.0,
                        trtllm_serve_stop_timeout_s=2.0,
                        trtllm_serve_library_path=("/opt/openmpi/lib", "/cuda/lib"),
                        trtllm_serve_env=(("CUDA_HOME", "/cuda"),),
                        trtllm_serve_config_path="/models/gemma4-trtllm.yaml",
                        trtllm_serve_reasoning_parser="gemma4",
                        trtllm_serve_tool_parser="gemma4",
                        trtllm_serve_extra_args=("--max_batch_size", "4"),
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
            if request.full_url == "http://127.0.0.1:18091/health":
                captured["health_timeout"] = timeout
                return FakeResponse({"status": "ok"})
            captured["chat_url"] = request.full_url
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
            mock.patch.object(
                trtllm_serve_module.subprocess,
                "Popen",
                side_effect=fake_popen,
            ),
            mock.patch.object(
                trtllm_serve_module,
                "urlopen",
                side_effect=fake_urlopen,
            ),
            mock.patch.object(trtllm_serve_module.os, "killpg") as killpg,
        ):
            engine = trtllm_serve_module.TrtllmServeEngine(settings)
            result = engine.complete(
                ResponseRequest(
                    model="gemma4",
                    input=[
                        TextContent(text="Describe this."),
                        ImageContent(
                            image_url=ImageUrlSpec(
                                url="data:image/png;base64,abc",
                            )
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
        self.assertEqual(command[0], "/opt/trtllm/bin/trtllm-serve")
        self.assertEqual(command[1], "serve")
        self.assertEqual(command[2], "/models/nvidia/Gemma-4-26B-A4B-NVFP4")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18091")
        self.assertEqual(
            command[command.index("--served_model_name") + 1],
            "gemma-local",
        )
        self.assertEqual(
            command[command.index("--config") + 1],
            "/models/gemma4-trtllm.yaml",
        )
        self.assertEqual(
            command[command.index("--reasoning_parser") + 1],
            "gemma4",
        )
        self.assertEqual(command[command.index("--tool_parser") + 1], "gemma4")
        self.assertIn("--trust_remote_code", command)
        self.assertIn("--no-telemetry", command)
        self.assertEqual(command[command.index("--max_batch_size") + 1], "4")
        popen_kwargs = captured["popen_kwargs"]
        self.assertTrue(popen_kwargs["start_new_session"])
        self.assertEqual(popen_kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(popen_kwargs["stdout"].closed)
        self.assertTrue(popen_kwargs["env"]["PATH"].startswith("/opt/trtllm/bin"))
        self.assertIn("/cuda/bin", popen_kwargs["env"]["PATH"].split(":"))
        self.assertTrue(
            popen_kwargs["env"]["LD_LIBRARY_PATH"].startswith(
                "/opt/openmpi/lib:/cuda/lib"
            )
        )
        self.assertEqual(popen_kwargs["env"]["CUDA_HOME"], "/cuda")
        self.assertEqual(popen_kwargs["env"]["PYTHONUNBUFFERED"], "1")
        self.assertEqual(captured["health_timeout"], 1.0)
        self.assertEqual(
            captured["chat_url"],
            "http://127.0.0.1:18091/v1/chat/completions",
        )
        self.assertEqual(captured["chat_timeout"], 12.5)
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
        killpg.assert_called_once_with(4242, signal.SIGTERM)

    def test_subprocess_env_unbuffers_output_without_other_overrides(self) -> None:
        settings = ModelSettings(
            model_path=None,
            backend="trtllm_serve",
        )

        env = trtllm_serve_module.TrtllmServeEngine._subprocess_env(settings)

        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    def test_close_kills_process_group_after_timeout(self) -> None:
        process = FakeProcess()
        process.wait = mock.Mock(
            side_effect=[
                subprocess.TimeoutExpired(cmd="trtllm-serve", timeout=2.0),
                -9,
            ]
        )
        runtime = trtllm_serve_module.TrtllmServeModelRuntime(
            config=mock.Mock(),
            process=process,
            base_url="http://127.0.0.1:18091/v1",
            health_url="http://127.0.0.1:18091/health",
            remote_model="gemma-local",
            timeout_s=12.5,
            stop_timeout_s=2.0,
            output_log=tempfile.TemporaryFile(),
        )

        with mock.patch.object(trtllm_serve_module.os, "killpg") as killpg:
            runtime.close()

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4242, signal.SIGTERM),
                mock.call(4242, signal.SIGKILL),
            ],
        )
        self.assertTrue(runtime.output_log.closed)

    def test_close_escalates_process_group_after_leader_exit(self) -> None:
        process = FakeProcess()
        process.return_code = 1
        process.wait = mock.Mock()
        runtime = trtllm_serve_module.TrtllmServeModelRuntime(
            config=mock.Mock(),
            process=process,
            base_url="http://127.0.0.1:18091/v1",
            health_url="http://127.0.0.1:18091/health",
            remote_model="gemma-local",
            timeout_s=12.5,
            stop_timeout_s=2.0,
            output_log=tempfile.TemporaryFile(),
        )

        with (
            mock.patch.object(trtllm_serve_module.os, "killpg") as killpg,
            mock.patch.object(
                trtllm_serve_module.time,
                "monotonic",
                side_effect=[10.0, 10.0, 12.0],
            ),
            mock.patch.object(trtllm_serve_module.time, "sleep") as sleep,
        ):
            runtime.close()
            runtime.close()

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4242, signal.SIGTERM),
                mock.call(4242, 0),
                mock.call(4242, signal.SIGKILL),
            ],
        )
        sleep.assert_called_once_with(0.1)
        process.wait.assert_not_called()
        self.assertTrue(runtime.output_log.closed)

    def test_close_does_not_raise_when_process_survives_sigkill(self) -> None:
        process = FakeProcess()
        process.wait = mock.Mock(
            side_effect=[
                subprocess.TimeoutExpired(cmd="trtllm-serve", timeout=2.0),
                subprocess.TimeoutExpired(cmd="trtllm-serve", timeout=5.0),
            ]
        )
        runtime = trtllm_serve_module.TrtllmServeModelRuntime(
            config=mock.Mock(),
            process=process,
            base_url="http://127.0.0.1:18091/v1",
            health_url="http://127.0.0.1:18091/health",
            remote_model="gemma-local",
            timeout_s=12.5,
            stop_timeout_s=2.0,
            output_log=tempfile.TemporaryFile(),
        )

        with (
            mock.patch.object(trtllm_serve_module.os, "killpg") as killpg,
            self.assertLogs(trtllm_serve_module.LOGGER, level="WARNING") as logs,
        ):
            runtime.close()

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4242, signal.SIGTERM),
                mock.call(4242, signal.SIGKILL),
            ],
        )
        self.assertIn("did not exit after SIGKILL", logs.output[0])
        self.assertTrue(runtime.output_log.closed)

    def test_startup_exit_includes_process_output_tail(self) -> None:
        process = FakeProcess()
        process.return_code = 17
        output_log = tempfile.TemporaryFile()
        output_log.write(b"engine initialized\naddress already in use\n")
        output_log.flush()
        runtime = trtllm_serve_module.TrtllmServeModelRuntime(
            config=mock.Mock(),
            process=process,
            base_url="http://127.0.0.1:18091/v1",
            health_url="http://127.0.0.1:18091/health",
            remote_model="gemma-local",
            timeout_s=12.5,
            stop_timeout_s=2.0,
            output_log=output_log,
        )
        self.addCleanup(output_log.close)
        engine = object.__new__(trtllm_serve_module.TrtllmServeEngine)

        with self.assertRaises(RuntimeError) as exc_info:
            engine._wait_until_ready(runtime, 1.0)

        self.assertIn("exited during startup with code 17", str(exc_info.exception))
        self.assertIn("address already in use", str(exc_info.exception))

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
            trtllm_serve_module.BackendExecutionError
        ) as exc_info:
            trtllm_serve_module.TrtllmServeEngine._extract_text(payload)

        self.assertEqual(
            exc_info.exception.code,
            "trtllm_serve_incomplete_response",
        )


if __name__ == "__main__":
    unittest.main()
