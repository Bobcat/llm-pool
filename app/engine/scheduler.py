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
    loaded_replicas: int


@dataclass(frozen=True)
class ReplicaRegistration:
    replica_id: str
    complete_fn: Callable[[ResponseRequest], EngineResult]
    runtime_capability: int = 1


@dataclass
class SchedulerJob:
    request: ResponseRequest
    result_future: Future[EngineResult]
    enqueued_at: float


@dataclass
class _ReplicaSnapshot:
    replica_id: str
    runtime_inflight: int
    effective_target_inflight: int
    accepting_new_requests: bool


class _ReplicaExecutor:
    def __init__(
        self,
        *,
        public_model_name: str,
        replica_id: str,
        complete_fn: Callable[[ResponseRequest], EngineResult],
        configured_target_inflight: int,
        runtime_capability: int,
        notify_released: Callable[[], None],
    ) -> None:
        self.public_model_name = str(public_model_name)
        self.replica_id = str(replica_id)
        self._complete_fn = complete_fn
        self._configured_target_inflight = max(1, int(configured_target_inflight))
        self._runtime_capability = max(1, int(runtime_capability))
        self._effective_target_inflight = min(self._configured_target_inflight, self._runtime_capability)
        self._notify_released = notify_released
        self._accepting_new_requests = True
        self._runtime_inflight = 0
        self._cond = threading.Condition()

    def snapshot(self) -> _ReplicaSnapshot:
        with self._cond:
            return _ReplicaSnapshot(
                replica_id=self.replica_id,
                runtime_inflight=self._runtime_inflight,
                effective_target_inflight=self._effective_target_inflight,
                accepting_new_requests=self._accepting_new_requests,
            )

    def submit(self, job: SchedulerJob, *, dequeued_at: float) -> None:
        with self._cond:
            if not self._accepting_new_requests or self._runtime_inflight >= self._effective_target_inflight:
                raise ModelStateError(self.public_model_name, "model_unloading")
            self._runtime_inflight += 1
        worker = threading.Thread(
            target=self._run_job,
            kwargs={"job": job, "dequeued_at": dequeued_at},
            name=f"llm-pool-replica-{self.replica_id}",
            daemon=True,
        )
        worker.start()

    def begin_shutdown(self) -> None:
        with self._cond:
            self._accepting_new_requests = False
            self._cond.notify_all()

    def join(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._runtime_inflight > 0:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining == 0.0:
                    break
                self._cond.wait(timeout=remaining)

    def _run_job(self, *, job: SchedulerJob, dequeued_at: float) -> None:
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
            self._notify_released()

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


class LoadedModelExecutor:
    def __init__(
        self,
        *,
        model_name: str,
        replicas: list[ReplicaRegistration],
        configured_target_inflight: int,
    ) -> None:
        self.model_name = str(model_name)
        self._configured_target_inflight = max(1, int(configured_target_inflight))
        self._pending_jobs: deque[SchedulerJob] = deque()
        self._accepting_new_requests = True
        self._stop_requested = False
        self._round_robin_index = 0
        self._cond = threading.Condition()
        self._replicas = [
            _ReplicaExecutor(
                public_model_name=self.model_name,
                replica_id=registration.replica_id,
                complete_fn=registration.complete_fn,
                configured_target_inflight=self._configured_target_inflight,
                runtime_capability=registration.runtime_capability,
                notify_released=self._notify_replica_released,
            )
            for registration in replicas
        ]
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
        replica_snapshots = [replica.snapshot() for replica in self._replicas]
        runtime_inflight = sum(snapshot.runtime_inflight for snapshot in replica_snapshots)
        effective_target_inflight = min(
            (snapshot.effective_target_inflight for snapshot in replica_snapshots),
            default=min(self._configured_target_inflight, 1),
        )
        with self._cond:
            return ExecutorSnapshot(
                queue_depth=len(self._pending_jobs),
                runtime_inflight=runtime_inflight,
                configured_target_inflight=self._configured_target_inflight,
                effective_target_inflight=effective_target_inflight,
                accepting_new_requests=self._accepting_new_requests,
                loaded_replicas=len(replica_snapshots),
            )

    def begin_shutdown(self) -> None:
        cancelled_jobs: list[SchedulerJob] = []
        with self._cond:
            self._accepting_new_requests = False
            self._stop_requested = True
            while self._pending_jobs:
                cancelled_jobs.append(self._pending_jobs.popleft())
            self._cond.notify_all()
        for replica in self._replicas:
            replica.begin_shutdown()
        for job in cancelled_jobs:
            self._set_future_exception(job.result_future, ModelStateError(self.model_name, "model_unloading"))

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)
        for replica in self._replicas:
            replica.join(timeout=timeout)

    def _run_loop(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._stop_requested and not self._pending_jobs and self._runtime_inflight_locked() == 0:
                        return
                    selected_replica = self._pick_replica_locked()
                    if self._pending_jobs and selected_replica is not None:
                        job = self._pending_jobs.popleft()
                        dequeued_at = time.perf_counter()
                        break
                    self._cond.wait()
            try:
                selected_replica.submit(job, dequeued_at=dequeued_at)
            except ModelStateError as exc:
                self._set_future_exception(job.result_future, exc)
                self._notify_replica_released()

    def _pick_replica_locked(self) -> _ReplicaExecutor | None:
        if not self._replicas:
            return None
        snapshots = [replica.snapshot() for replica in self._replicas]
        eligible_indices = [
            index
            for index, snapshot in enumerate(snapshots)
            if snapshot.accepting_new_requests and snapshot.runtime_inflight < snapshot.effective_target_inflight
        ]
        if not eligible_indices:
            return None
        lowest_inflight = min(snapshots[index].runtime_inflight for index in eligible_indices)
        candidate_indices = [index for index in eligible_indices if snapshots[index].runtime_inflight == lowest_inflight]
        total_replicas = len(self._replicas)
        for offset in range(total_replicas):
            index = (self._round_robin_index + offset) % total_replicas
            if index in candidate_indices:
                self._round_robin_index = (index + 1) % total_replicas
                return self._replicas[index]
        return self._replicas[candidate_indices[0]]

    def _runtime_inflight_locked(self) -> int:
        return sum(replica.snapshot().runtime_inflight for replica in self._replicas)

    def _notify_replica_released(self) -> None:
        with self._cond:
            self._cond.notify_all()

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
        replicas: list[ReplicaRegistration],
        configured_target_inflight: int,
    ) -> LoadedModelExecutor:
        executor = LoadedModelExecutor(
            model_name=model_name,
            replicas=replicas,
            configured_target_inflight=configured_target_inflight,
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
