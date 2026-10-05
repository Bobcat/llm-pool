from __future__ import annotations

import importlib.util
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import socket
import threading
import time
import unittest
from unittest import mock

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.config import AppSettings
    from app.config import DecodingDefaults
    from app.config import EngineSettings
    from app.config import ModelSettings
    import app.engine.vllm_serve as vllm_serve_module
    from app.engine.scheduler import CancellationToken
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


class FakeStreamingResponse:
    def __init__(self, events: list[dict[str, object] | str]) -> None:
        self._events = events
        self.closed = False

    def __iter__(self):
        for event in self._events:
            data = event if isinstance(event, str) else json.dumps(event)
            yield f"data: {data}\n".encode("utf-8")
            yield b"\n"

    def close(self) -> None:
        self.closed = True


class BlockingStreamingResponse:
    def __init__(self, first_event: dict[str, object]) -> None:
        self._first_event = first_event
        self.first_sent = threading.Event()
        self.closed = threading.Event()

    def __iter__(self):
        yield f"data: {json.dumps(self._first_event)}\n".encode("utf-8")
        yield b"\n"
        self.first_sent.set()
        self.closed.wait(timeout=2.0)
        raise OSError("response closed")

    def close(self) -> None:
        self.closed.set()


class RawStreamingResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        yield from self._lines

    def close(self) -> None:
        pass


class TimeoutStreamingResponse(FakeStreamingResponse):
    def __iter__(self):
        yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n'
        yield b"\n"
        raise socket.timeout("idle timeout")


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
    @staticmethod
    def _engine():
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults()
        engine._models = {
            "gemma4": mock.Mock(
                remote_model="gemma4",
                config=ModelSettings(model_path=None, backend="vllm_serve"),
            )
        }
        return engine

    @staticmethod
    def _fake_open_stream(upstream, captured: dict[str, object] | None = None):
        connection = mock.Mock()

        def open_stream(_runtime, payload, cancellation):
            if captured is not None:
                captured["payload"] = payload
            cancel = upstream.close
            cancellation.set_callback(cancel)
            return upstream, connection, cancel

        return open_stream

    def test_stream_abort_shuts_down_socket_before_closing_connection(self) -> None:
        connection = mock.Mock()
        upstream_socket = mock.Mock(spec=socket.socket)
        operations: list[str] = []
        upstream_socket.shutdown.side_effect = lambda how: operations.append(
            f"shutdown:{how}"
        )
        connection.close.side_effect = lambda: operations.append("close")

        vllm_serve_module.VllmServeEngine._abort_stream_connection(
            connection,
            upstream_socket,
        )

        self.assertEqual(operations, [f"shutdown:{socket.SHUT_RDWR}", "close"])

    def test_stream_abort_logs_when_connection_has_no_socket(self) -> None:
        connection = mock.Mock()

        with self.assertLogs("llm_pool.engine", level="WARNING") as logs:
            vllm_serve_module.VllmServeEngine._abort_stream_connection(
                connection,
                None,
            )

        connection.close.assert_called_once_with()
        self.assertIn("connection has no active socket", "\n".join(logs.output))

    def test_stream_forwards_incremental_multimodal_deltas_and_usage(self) -> None:
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults()
        runtime = mock.Mock(
            remote_model="gemma4",
            config=ModelSettings(
                model_path=None,
                backend="vllm_serve",
                prompt_format="gemma4_template",
                enable_thinking=False,
            ),
        )
        engine._models = {"gemma4": runtime}
        upstream = FakeStreamingResponse(
            [
                {
                    "id": "chat-1",
                    "model": "gemma4",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 0},
                },
                {
                    "id": "chat-1",
                    "choices": [{"index": 0, "delta": {"reasoning": "Check."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                },
                {
                    "id": "chat-1",
                    "choices": [{"index": 0, "delta": {"content": "Hel"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
                {
                    "id": "chat-1",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "lo"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
                {
                    "id": "chat-1",
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
                "[DONE]",
            ]
        )
        captured: dict[str, object] = {}

        events = []
        request = ResponseRequest(
            model="gemma4",
            input=[
                TextContent(text="Describe this."),
                ImageContent(image_url=ImageUrlSpec(url="data:image/png;base64,abc")),
            ],
            thinking="disabled",
            decoding=DecodingParams(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                max_tokens=4096,
            ),
        )
        def emit(event) -> bool:
            events.append(event)
            return True

        with mock.patch.object(
            engine,
            "_open_stream",
            side_effect=self._fake_open_stream(upstream, captured),
        ):
            result = engine.stream(request, emit, CancellationToken())

        payload = captured["payload"]
        self.assertTrue(payload["stream"])
        self.assertEqual(
            payload["stream_options"],
            {"include_usage": True, "continuous_usage_stats": True},
        )
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(
            [(event.type, event.delta) for event in events],
            [
                ("reasoning_text.delta", "Check."),
                ("output_text.delta", "Hel"),
                ("output_text.delta", "lo"),
            ],
        )
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.reasoning_text, "Check.")
        self.assertEqual(result.metrics.engine_prompt_tokens, 10)
        self.assertEqual(result.metrics.engine_output_tokens, 3)
        self.assertEqual(result.metrics.engine_finish_reason, "stop")
        self.assertEqual(
            result.metadata["upstream_response"]["choices"][0]["finish_reason"],
            "stop",
        )
        self.assertTrue(upstream.closed)

    def test_cancelling_stream_closes_upstream_and_returns_partial_metrics(self) -> None:
        engine = self._engine()
        upstream = BlockingStreamingResponse(
            {
                "id": "chat-1",
                "choices": [{"index": 0, "delta": {"content": "partial"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2},
            }
        )
        cancellation = CancellationToken()
        result_holder: dict[str, object] = {}

        def run_stream() -> None:
            with mock.patch.object(
                engine,
                "_open_stream",
                side_effect=self._fake_open_stream(upstream),
            ):
                result_holder["result"] = engine.stream(
                    ResponseRequest(model="gemma4", input="Hello", stream=True),
                    lambda event: True,
                    cancellation,
                )

        worker = threading.Thread(target=run_stream)
        worker.start()
        self.assertTrue(upstream.first_sent.wait(timeout=1.0))

        cancellation.cancel()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertTrue(upstream.closed.is_set())
        result = result_holder["result"]
        self.assertEqual(result.text, "partial")
        self.assertEqual(result.metrics.engine_prompt_tokens, 8)
        self.assertEqual(result.metrics.engine_output_tokens, 2)
        self.assertEqual(result.metrics.engine_finish_reason, "cancelled")

    def test_stream_deltas_concatenate_to_trimmed_output_text(self) -> None:
        engine = self._engine()
        upstream = FakeStreamingResponse(
            [
                {"choices": [{"delta": {"content": "\n\n"}}]},
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "\n"}}]},
                {"choices": [{"delta": {"content": "world"}}]},
                {
                    "choices": [
                        {
                            "delta": {"content": " \n"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                "[DONE]",
            ]
        )
        deltas: list[str] = []

        with mock.patch.object(
            engine,
            "_open_stream",
            side_effect=self._fake_open_stream(upstream),
        ):
            result = engine.stream(
                ResponseRequest(model="gemma4", input="Hello", stream=True),
                lambda event: deltas.append(event.delta) or True,
                CancellationToken(),
            )

        self.assertEqual("".join(deltas), result.text)
        self.assertEqual(result.text, "Hello \nworld")

    def test_stream_reasoning_only_matches_non_streaming_acceptance_rule(self) -> None:
        cases = (
            ("enabled", "length", False),
            ("enabled", "stop", True),
            ("default", "length", True),
            ("default", "stop", True),
        )
        for thinking, finish_reason, should_raise in cases:
            with self.subTest(thinking=thinking, finish_reason=finish_reason):
                engine = self._engine()
                upstream = FakeStreamingResponse(
                    [
                        {
                            "choices": [
                                {
                                    "delta": {"reasoning": "Still thinking."},
                                    "finish_reason": finish_reason,
                                }
                            ]
                        },
                        "[DONE]",
                    ]
                )
                request = ResponseRequest(
                    model="gemma4",
                    input="Hello",
                    stream=True,
                    thinking=thinking,
                )
                patch = mock.patch.object(
                    engine,
                    "_open_stream",
                    side_effect=self._fake_open_stream(upstream),
                )
                if should_raise:
                    with patch, self.assertRaises(vllm_serve_module.BackendExecutionError):
                        engine.stream(request, lambda event: True, CancellationToken())
                else:
                    with patch:
                        result = engine.stream(
                            request,
                            lambda event: True,
                            CancellationToken(),
                        )
                    self.assertEqual(result.text, "")
                    self.assertEqual(result.reasoning_text, "Still thinking.")

    def test_cancelling_after_reasoning_returns_partial_cancelled_result(self) -> None:
        engine = self._engine()
        upstream = FakeStreamingResponse(
            [
                {
                    "choices": [{"delta": {"reasoning": "Still thinking."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
                "[DONE]",
            ]
        )
        cancellation = CancellationToken()

        def emit(event) -> bool:
            cancellation.cancel()
            return True

        with mock.patch.object(
            engine,
            "_open_stream",
            side_effect=self._fake_open_stream(upstream),
        ):
            result = engine.stream(
                ResponseRequest(
                    model="gemma4",
                    input="Hello",
                    stream=True,
                    thinking="enabled",
                ),
                emit,
                cancellation,
            )

        self.assertEqual(result.text, "")
        self.assertEqual(result.reasoning_text, "Still thinking.")
        self.assertEqual(result.metrics.engine_prompt_tokens, 10)
        self.assertEqual(result.metrics.engine_output_tokens, 3)
        self.assertEqual(result.metrics.engine_finish_reason, "cancelled")

    def test_abort_before_request_write_does_not_reconnect(self) -> None:
        engine = self._engine()
        runtime = engine._models["gemma4"]
        runtime.base_url = "http://127.0.0.1:12345/v1"
        runtime.timeout_s = 5.0
        runtime.api_key = None
        cancellation = CancellationToken()

        class RaceConnection:
            def __init__(self) -> None:
                self.auto_open = 1
                self.connect_count = 0
                self.request_delivered = False
                self.sock = None

            def connect(self) -> None:
                self.connect_count += 1
                self.sock = mock.Mock(spec=socket.socket)

            def close(self) -> None:
                self.sock = None

            def request(self, *args, **kwargs) -> None:
                del args, kwargs
                cancellation.cancel()
                if self.sock is None:
                    if self.auto_open:
                        self.connect()
                    else:
                        raise OSError("socket is closed")
                self.request_delivered = True

            def getresponse(self):
                raise AssertionError("cancelled request reached getresponse()")

        connection = RaceConnection()
        with mock.patch.object(
            vllm_serve_module,
            "HTTPConnection",
            return_value=connection,
        ):
            result = engine.stream(
                ResponseRequest(model="gemma4", input="Hello", stream=True),
                lambda event: True,
                cancellation,
            )

        self.assertEqual(connection.connect_count, 1)
        self.assertFalse(connection.request_delivered)
        self.assertEqual(result.metrics.engine_finish_reason, "cancelled")

    def test_stream_maps_upstream_error_event(self) -> None:
        engine = self._engine()
        upstream = FakeStreamingResponse(
            [{"error": {"message": "context length exceeded"}}, "[DONE]"]
        )

        with (
            mock.patch.object(
                engine,
                "_open_stream",
                side_effect=self._fake_open_stream(upstream),
            ),
            self.assertRaises(vllm_serve_module.BackendExecutionError) as exc_info,
        ):
            engine.stream(
                ResponseRequest(model="gemma4", input="Hello", stream=True),
                lambda event: True,
                CancellationToken(),
            )

        self.assertEqual(exc_info.exception.code, "vllm_serve_error")
        self.assertEqual(exc_info.exception.status_code, 502)
        self.assertEqual(exc_info.exception.message, "context length exceeded")

    def test_stream_rejects_malformed_sse_payloads(self) -> None:
        cases = (
            ("invalid JSON", ["{"]),
            ("non-object JSON", ["[]"]),
            ("non-object choice", [{"choices": ["bad"]}]),
            (
                "missing finish reason",
                [{"choices": [{"delta": {"content": "complete"}}]}, "[DONE]"],
            ),
            (
                "missing final event",
                [{"choices": [{"delta": {"content": "partial"}, "finish_reason": "stop"}]}],
            ),
        )
        for label, events in cases:
            with self.subTest(label=label):
                engine = self._engine()
                upstream = FakeStreamingResponse(events)
                with (
                    mock.patch.object(
                        engine,
                        "_open_stream",
                        side_effect=self._fake_open_stream(upstream),
                    ),
                    self.assertRaises(vllm_serve_module.BackendExecutionError) as exc_info,
                ):
                    engine.stream(
                        ResponseRequest(model="gemma4", input="Hello", stream=True),
                        lambda event: True,
                        CancellationToken(),
                    )
                self.assertEqual(
                    exc_info.exception.code,
                    "vllm_serve_response_parse_failure",
                )

    def test_stream_rejects_invalid_utf8(self) -> None:
        engine = self._engine()
        upstream = RawStreamingResponse([b"data: \xff\n", b"\n"])

        with (
            mock.patch.object(
                engine,
                "_open_stream",
                side_effect=self._fake_open_stream(upstream),
            ),
            self.assertRaises(vllm_serve_module.BackendExecutionError) as exc_info,
        ):
            engine.stream(
                ResponseRequest(model="gemma4", input="Hello", stream=True),
                lambda event: True,
                CancellationToken(),
            )

        self.assertEqual(
            exc_info.exception.code,
            "vllm_serve_response_parse_failure",
        )

    def test_stream_maps_midstream_idle_timeout(self) -> None:
        engine = self._engine()
        upstream = TimeoutStreamingResponse([])

        with (
            mock.patch.object(
                engine,
                "_open_stream",
                side_effect=self._fake_open_stream(upstream),
            ),
            self.assertRaises(vllm_serve_module.BackendExecutionError) as exc_info,
        ):
            engine.stream(
                ResponseRequest(model="gemma4", input="Hello", stream=True),
                lambda event: True,
                CancellationToken(),
            )

        self.assertEqual(exc_info.exception.code, "vllm_serve_timeout")

    def test_cancellation_interrupts_wait_for_response_headers(self) -> None:
        request_started = threading.Event()
        release_handler = threading.Event()

        class DelayedHeadersHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(content_length)
                request_started.set()
                release_handler.wait(timeout=2.0)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                except OSError:
                    pass

            def log_message(self, format, *args) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedHeadersHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        try:
            engine = self._engine()
            runtime = engine._models["gemma4"]
            runtime.base_url = f"http://127.0.0.1:{server.server_port}/v1"
            runtime.timeout_s = 5.0
            runtime.api_key = None
            cancellation = CancellationToken()
            result_holder: dict[str, object] = {}

            def run_stream() -> None:
                result_holder["result"] = engine.stream(
                    ResponseRequest(model="gemma4", input="Hello", stream=True),
                    lambda event: True,
                    cancellation,
                )

            worker = threading.Thread(target=run_stream)
            worker.start()
            self.assertTrue(request_started.wait(timeout=1.0))

            started = time.perf_counter()
            cancellation.cancel()
            worker.join(timeout=1.0)

            self.assertFalse(worker.is_alive())
            self.assertLess(time.perf_counter() - started, 1.0)
            self.assertEqual(
                result_holder["result"].metrics.engine_finish_reason,
                "cancelled",
            )
        finally:
            release_handler.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_cancellation_interrupts_real_blocked_stream_reader(self) -> None:
        first_chunk_sent = threading.Event()
        release_handler = threading.Event()

        class StalledStreamHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(content_length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                chunk = (
                    b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
                )
                self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
                self.wfile.write(chunk + b"\r\n")
                self.wfile.flush()
                first_chunk_sent.set()
                release_handler.wait(timeout=2.0)

            def log_message(self, format, *args) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), StalledStreamHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        try:
            engine = self._engine()
            runtime = engine._models["gemma4"]
            runtime.base_url = f"http://127.0.0.1:{server.server_port}/v1"
            runtime.timeout_s = 5.0
            runtime.api_key = None
            cancellation = CancellationToken()
            delta_received = threading.Event()
            result_holder: dict[str, object] = {}

            def emit(event) -> bool:
                delta_received.set()
                return True

            def run_stream() -> None:
                result_holder["result"] = engine.stream(
                    ResponseRequest(model="gemma4", input="Hello", stream=True),
                    emit,
                    cancellation,
                )

            worker = threading.Thread(target=run_stream)
            worker.start()
            self.assertTrue(first_chunk_sent.wait(timeout=1.0))
            self.assertTrue(delta_received.wait(timeout=1.0))

            started = time.perf_counter()
            cancellation.cancel()
            worker.join(timeout=1.0)

            self.assertFalse(worker.is_alive())
            self.assertLess(time.perf_counter() - started, 1.0)
            self.assertEqual(result_holder["result"].text, "partial")
            self.assertEqual(
                result_holder["result"].metrics.engine_finish_reason,
                "cancelled",
            )
        finally:
            release_handler.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_stream_maps_upstream_http_error(self) -> None:
        class RejectedStreamHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(content_length)
                body = json.dumps(
                    {"error": {"message": "context length exceeded"}}
                ).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), RejectedStreamHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        try:
            engine = self._engine()
            runtime = engine._models["gemma4"]
            runtime.base_url = f"http://127.0.0.1:{server.server_port}/v1"
            runtime.timeout_s = 5.0
            runtime.api_key = None

            with self.assertRaises(vllm_serve_module.BackendExecutionError) as exc_info:
                engine.stream(
                    ResponseRequest(model="gemma4", input="Hello", stream=True),
                    lambda event: True,
                    CancellationToken(),
                )

            self.assertEqual(exc_info.exception.code, "vllm_serve_invalid_request")
            self.assertEqual(exc_info.exception.status_code, 502)
            self.assertEqual(exc_info.exception.message, "context length exceeded")
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_rejects_max_num_seqs_in_extra_args(self) -> None:
        for extra_args in (
            ("--max-num-seqs", "8"),
            ("--max-num-seqs=8",),
            ("--max_num_seqs", "8"),
            ("--max_num_seqs=8",),
        ):
            with self.subTest(extra_args=extra_args):
                settings = ModelSettings(
                    model_path=None,
                    backend="vllm_serve",
                    vllm_model="/models/gemma4",
                    vllm_serve_extra_args=extra_args,
                )

                with self.assertRaisesRegex(ValueError, "controlled by target_inflight"):
                    vllm_serve_module.VllmServeEngine.__new__(
                        vllm_serve_module.VllmServeEngine
                    )._command(
                        settings=settings,
                        model_ref="/models/gemma4",
                        host="127.0.0.1",
                        port=18090,
                        remote_model="gemma4",
                    )

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

    def test_gemma4_thinking_token_budget_is_forwarded(self) -> None:
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults()
        payload = engine._chat_completions_payload(
            runtime=mock.Mock(
                remote_model="gemma4",
                config=ModelSettings(
                    model_path=None,
                    backend="vllm_serve",
                    prompt_format="gemma4_template",
                ),
            ),
            request=ResponseRequest(
                model="gemma4",
                input="Write a story",
                thinking="enabled",
                thinking_token_budget=256,
            ),
            decoding=engine._resolve_decoding(DecodingParams(max_tokens=512)),
        )

        self.assertEqual(payload["thinking_token_budget"], 256)

    def test_gemma4_default_thinking_uses_model_configuration(self) -> None:
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults()
        runtime = mock.Mock(
            remote_model="gemma4",
            config=ModelSettings(
                model_path=None,
                backend="vllm_serve",
                prompt_format="gemma4_template",
                enable_thinking=False,
            ),
        )

        payload = engine._chat_completions_payload(
            runtime=runtime,
            request=ResponseRequest(model="gemma4", input="Hello"),
            decoding=engine._resolve_decoding(DecodingParams()),
        )

        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_gemma4_reasoning_effort_enables_thinking_and_accepts_partial_reasoning(self) -> None:
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults()
        runtime = mock.Mock(
            remote_model="gemma4",
            config=ModelSettings(
                model_path=None,
                backend="vllm_serve",
                prompt_format="gemma4_template",
                enable_thinking=False,
                reasoning_efforts=("none", "low"),
                thinking_token_budget_max=512,
            ),
        )
        engine._models = {"gemma4": runtime}
        upstream = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": None, "reasoning": "Still thinking."},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 128},
        }

        with mock.patch.object(engine, "_post_json", return_value=upstream) as post:
            result = engine.complete(
                ResponseRequest(
                    model="gemma4",
                    input="Write a story",
                    reasoning_effort="low",
                    thinking_token_budget=128,
                )
            )

        self.assertEqual(post.call_args.args[1]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(post.call_args.args[1]["reasoning_effort"], "low")
        self.assertEqual(result.text, "")
        self.assertEqual(result.reasoning_text, "Still thinking.")

    def test_gemma4_thinking_uses_template_kwargs_and_exposes_reasoning(self) -> None:
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults()
        runtime = mock.Mock(
            remote_model="gemma4",
            config=ModelSettings(
                model_path=None,
                backend="vllm_serve",
                prompt_format="gemma4_template",
            ),
        )
        engine._models = {"gemma4": runtime}
        upstream = {
            "choices": [{"message": {"content": "391", "reasoning": "Calculate first."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7},
        }
        with mock.patch.object(engine, "_post_json", return_value=upstream) as post:
            result = engine.complete(
                ResponseRequest(model="gemma4", input="17 * 23?", thinking="enabled")
            )

        self.assertEqual(
            post.call_args.args[1]["chat_template_kwargs"],
            {"enable_thinking": True},
        )
        self.assertEqual(result.text, "391")
        self.assertEqual(result.reasoning_text, "Calculate first.")

        with mock.patch.object(engine, "_post_json", return_value=upstream) as post:
            engine.complete(ResponseRequest(model="gemma4", input="17 * 23?", thinking="disabled"))
        self.assertEqual(
            post.call_args.args[1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_gemma4_keeps_reasoning_when_token_limit_prevents_final_answer(self) -> None:
        engine = vllm_serve_module.VllmServeEngine.__new__(
            vllm_serve_module.VllmServeEngine
        )
        engine.decoding_defaults = DecodingDefaults()
        engine._models = {
            "gemma4": mock.Mock(
                remote_model="gemma4",
                config=ModelSettings(
                    model_path=None,
                    backend="vllm_serve",
                    prompt_format="gemma4_template",
                ),
            )
        }
        upstream = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": None, "reasoning": "Still thinking."},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 256},
        }
        with mock.patch.object(engine, "_post_json", return_value=upstream):
            result = engine.complete(
                ResponseRequest(model="gemma4", input="Write a story", thinking="enabled")
            )

        self.assertEqual(result.text, "")
        self.assertEqual(result.reasoning_text, "Still thinking.")
        self.assertEqual(result.metrics.engine_finish_reason, "length")

        with self.assertRaises(vllm_serve_module.BackendExecutionError):
            with mock.patch.object(engine, "_post_json", return_value=upstream):
                engine.complete(ResponseRequest(model="gemma4", input="Write a story"))

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
                        target_inflight=7,
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
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "7")
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
