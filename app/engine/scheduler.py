from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import threading
import time
from typing import Callable

from app.schemas import EngineResult
from app.schemas import ResponseMetrics
from app.schemas import ResponseRequest

from .common import ModelStateError


@dataclass(frozen=True)
class ExecutorSnapshot:
    queue_depth: int
    runtime_inflight: int
    configured_target_inflight: int
    effective_target_inflight: int
    accepting_new_requests: bool


@dataclass
class SchedulerJob:
    request: ResponseRequest
    result_future: Future[EngineResult]
    enqueued_at: float


class LoadedModelExecutor:
    def __init__(
        self,
        *,
        model_name: str,
        complete_fn: Callable[[ResponseRequest], EngineResult],
        configured_target_inflight: int,
        runtime_capability: int = 1,
    ) -> None:
        self.model_name = str(model_name)
        self._complete_fn = complete_fn
        self._configured_target_inflight = max(1, int(configured_target_inflight))
        self._runtime_capability = max(1, int(runtime_capability))
        self._effective_target_inflight = min(self._configured_target_inflight, self._runtime_capability)
        self._accepting_new_requests = True
        self._stop_requested = False
        self._runtime_inflight = 0
        self._pending_jobs: deque[SchedulerJob] = deque()
        self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"llm-pool-executor-{self.model_name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, request: ResponseRequest) -> Future[EngineResult]:
        result_future: Future[EngineResult] = Future()
        with self._cond:
            if not self._accepting_new_requests or self._stop_requested:
                raise ModelStateError(self.model_name, "model_unloading")
            self._pending_jobs.append(
                SchedulerJob(
                    request=request,
                    result_future=result_future,
                    enqueued_at=time.perf_counter(),
                )
            )
            self._cond.notify_all()
        return result_future

    def snapshot(self) -> ExecutorSnapshot:
        with self._cond:
            return ExecutorSnapshot(
                queue_depth=len(self._pending_jobs),
                runtime_inflight=self._runtime_inflight,
                configured_target_inflight=self._configured_target_inflight,
                effective_target_inflight=self._effective_target_inflight,
                accepting_new_requests=self._accepting_new_requests,
            )

    def begin_shutdown(self) -> None:
        cancelled_jobs: list[SchedulerJob] = []
        with self._cond:
            self._accepting_new_requests = False
            self._stop_requested = True
            while self._pending_jobs:
                cancelled_jobs.append(self._pending_jobs.popleft())
            self._cond.notify_all()
        for job in cancelled_jobs:
            self._set_future_exception(job.result_future, ModelStateError(self.model_name, "model_unloading"))

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._stop_requested and self._runtime_inflight == 0 and not self._pending_jobs:
                        return
                    if (
                        self._accepting_new_requests
                        and self._pending_jobs
                        and self._runtime_inflight < self._effective_target_inflight
                    ):
                        job = self._pending_jobs.popleft()
                        self._runtime_inflight += 1
                        dequeued_at = time.perf_counter()
                        break
                    self._cond.wait()
            try:
                backend_started_at = time.perf_counter()
                result = self._complete_fn(job.request)
                backend_finished_at = time.perf_counter()
                metrics_payload = (
                    result.metrics.model_dump()
                    if hasattr(result.metrics, "model_dump")
                    else result.metrics.dict()
                )
                metrics_payload["engine_queue_wait_ms"] = max(0.0, (dequeued_at - job.enqueued_at) * 1000.0)
                metrics_payload["backend_inference_wall_ms"] = max(
                    0.0,
                    (backend_finished_at - backend_started_at) * 1000.0,
                )
                result = EngineResult(
                    text=result.text,
                    metrics=ResponseMetrics(**metrics_payload),
                )
            except Exception as exc:
                self._set_future_exception(job.result_future, exc)
            else:
                self._set_future_result(job.result_future, result)
            finally:
                with self._cond:
                    if self._runtime_inflight > 0:
                        self._runtime_inflight -= 1
                    self._cond.notify_all()

    @staticmethod
    def _set_future_result(result_future: Future[EngineResult], result: EngineResult) -> None:
        try:
            result_future.set_result(result)
        except Exception:
            pass

    @staticmethod
    def _set_future_exception(result_future: Future[EngineResult], exc: Exception) -> None:
        try:
            result_future.set_exception(exc)
        except Exception:
            pass


class RuntimeScheduler:
    def __init__(self) -> None:
        self._executors: dict[str, LoadedModelExecutor] = {}
        self._lock = threading.RLock()

    def register(
        self,
        *,
        model_name: str,
        complete_fn: Callable[[ResponseRequest], EngineResult],
        configured_target_inflight: int,
        runtime_capability: int = 1,
    ) -> LoadedModelExecutor:
        executor = LoadedModelExecutor(
            model_name=model_name,
            complete_fn=complete_fn,
            configured_target_inflight=configured_target_inflight,
            runtime_capability=runtime_capability,
        )
        with self._lock:
            if model_name in self._executors:
                raise ValueError(f"executor already registered for model: {model_name}")
            self._executors[model_name] = executor
        executor.start()
        return executor

    def get(self, model_name: str) -> LoadedModelExecutor | None:
        with self._lock:
            return self._executors.get(model_name)

    def unregister(self, model_name: str) -> LoadedModelExecutor | None:
        with self._lock:
            return self._executors.pop(model_name, None)

    def close(self) -> None:
        with self._lock:
            executors = list(self._executors.values())
            self._executors.clear()
        for executor in executors:
            executor.begin_shutdown()
        for executor in executors:
            executor.join(timeout=1.0)
