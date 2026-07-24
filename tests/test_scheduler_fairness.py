from __future__ import annotations

from concurrent.futures import Future
import threading
import time
import unittest

from app.config import FairnessSettings
from app.engine.common import RequestAdmissionError
from app.engine.scheduler import _FairPendingQueue
from app.engine.scheduler import LoadedModelExecutor
from app.engine.scheduler import ReplicaRegistration
from app.schemas import EngineResult
from app.schemas import ResponseRequest


class FairPendingQueueTests(unittest.TestCase):
    def _queue(self, **overrides: object) -> _FairPendingQueue:
        settings = FairnessSettings(**overrides)
        return _FairPendingQueue(model_name="test-model", settings=settings)

    def _enqueue(
        self,
        queue: _FairPendingQueue,
        *,
        key: str | None,
        text: str,
        now: float = 0.0,
    ) -> None:
        queue.enqueue(
            request=ResponseRequest(model="test-model", input=text, fairness_key=key),
            result_future=Future[EngineResult](),
            now=now,
        )

    def test_preserves_fifo_order_within_one_key(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key="pdf", text="first")
        self._enqueue(queue, key="pdf", text="second")

        first = queue.pop_next(now=0.0)
        self.assertIsNotNone(first)
        queue.complete(first, service_ms=1.0, now=0.001)
        second = queue.pop_next(now=0.001)

        self.assertEqual(first.request.input, "first")
        self.assertEqual(second.request.input, "second")

    def test_equal_scores_rotate_deterministically_between_keys(self) -> None:
        queue = self._queue()
        for text in ("a1", "a2"):
            self._enqueue(queue, key="a", text=text)
        for text in ("b1", "b2"):
            self._enqueue(queue, key="b", text=text)

        selected: list[str | None] = []
        for index in range(4):
            job = queue.pop_next(now=float(index))
            selected.append(job.fairness_key)
            queue.complete(job, service_ms=0.0, now=float(index))

        self.assertEqual(selected, ["a", "b", "a", "b"])

    def test_weighted_service_prefers_key_with_more_weight(self) -> None:
        queue = self._queue(weights={"a": 2.0, "b": 1.0})
        for index in range(3):
            self._enqueue(queue, key="a", text=f"a{index}")
            self._enqueue(queue, key="b", text=f"b{index}")

        selected: list[str | None] = []
        for index in range(3):
            job = queue.pop_next(now=float(index))
            selected.append(job.fairness_key)
            queue.complete(job, service_ms=100.0, now=float(index))

        self.assertEqual(selected, ["a", "b", "a"])

    def test_weighted_long_run_tracks_slot_time_with_unequal_job_durations(self) -> None:
        queue = self._queue(weights={"a": 2.0, "b": 1.0})
        for index in range(2):
            self._enqueue(queue, key="a", text=f"a{index}")
            self._enqueue(queue, key="b", text=f"b{index}")

        service_by_key = {"a": 0.0, "b": 0.0}
        for _ in range(60):
            job = queue.pop_next(now=0.0)
            service_ms = 80.0 if job.fairness_key == "a" else 30.0
            service_by_key[job.fairness_key] += service_ms
            queue.complete(job, service_ms=service_ms, now=0.0)
            self._enqueue(
                queue,
                key=job.fairness_key,
                text=f"{job.fairness_key}-next",
            )

        normalized_a = service_by_key["a"] / 2.0
        normalized_b = service_by_key["b"]
        self.assertLessEqual(abs(normalized_a - normalized_b), 80.0)

    def test_active_elapsed_service_affects_next_selection(self) -> None:
        queue = self._queue(soft_max_inflight_per_key=2)
        self._enqueue(queue, key="a", text="a1")
        self._enqueue(queue, key="a", text="a2")
        self._enqueue(queue, key="b", text="b1")

        active = queue.pop_next(now=0.0)
        next_job = queue.pop_next(now=0.1)

        self.assertEqual(active.fairness_key, "a")
        self.assertEqual(next_job.fairness_key, "b")

    def test_soft_cap_is_borrowed_when_no_other_key_can_run(self) -> None:
        queue = self._queue(soft_max_inflight_per_key=1)
        self._enqueue(queue, key="a", text="a1")
        self._enqueue(queue, key="a", text="a2")

        first = queue.pop_next(now=0.0)
        second = queue.pop_next(now=0.1)

        self.assertEqual(first.fairness_key, "a")
        self.assertEqual(second.fairness_key, "a")

    def test_waiting_key_gets_next_slot_before_borrowing_key(self) -> None:
        queue = self._queue(soft_max_inflight_per_key=1)
        self._enqueue(queue, key="a", text="a1")
        self._enqueue(queue, key="a", text="a2")
        first = queue.pop_next(now=0.0)
        self._enqueue(queue, key="b", text="b1", now=0.01)

        second = queue.pop_next(now=0.02)

        self.assertEqual(first.fairness_key, "a")
        self.assertEqual(second.fairness_key, "b")

    def test_anonymous_and_keyed_requests_share_one_scheduler(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key=None, text="first")
        self._enqueue(queue, key=None, text="second")
        self._enqueue(queue, key="workbench", text="keyed")

        first = queue.pop_next(now=0.0)
        queue.complete(first, service_ms=0.0, now=0.001)
        keyed = queue.pop_next(now=0.001)
        queue.complete(keyed, service_ms=0.0, now=0.002)
        second = queue.pop_next(now=0.002)

        self.assertIsNone(first.fairness_key)
        self.assertEqual(keyed.fairness_key, "workbench")
        self.assertIsNone(second.fairness_key)
        self.assertEqual(second.request.input, "second")

    def test_new_key_starts_at_active_minimum_score(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key="a", text="a1")
        self._enqueue(queue, key="a", text="a2")
        first = queue.pop_next(now=0.0)
        queue.complete(first, service_ms=100.0, now=0.1)

        self._enqueue(queue, key="b", text="b1", now=0.1)

        self.assertEqual(queue.score("a", now=0.1), 100.0)
        self.assertEqual(queue.score("b", now=0.1), 100.0)

    def test_idle_state_expires_without_score_decay(self) -> None:
        queue = self._queue(idle_state_ttl_s=10.0)
        self._enqueue(queue, key="a", text="a1")
        first = queue.pop_next(now=0.0)
        queue.complete(first, service_ms=100.0, now=1.0)
        self.assertEqual(queue.score("a", now=9.0), 100.0)

        self.assertIsNone(queue.score("a", now=11.0))

    def test_per_key_queue_limit_has_stable_429_error(self) -> None:
        queue = self._queue(max_pending_per_key=1, max_pending_per_executor=4)
        self._enqueue(queue, key="pdf", text="first")

        with self.assertRaises(RequestAdmissionError) as exc_info:
            self._enqueue(queue, key="pdf", text="second")

        self.assertEqual(exc_info.exception.code, "fairness_key_queue_full")
        self.assertEqual(exc_info.exception.status_code, 429)
        self.assertEqual(queue.rejected_count, 1)
        snapshot = queue.snapshots(now=0.0)[0]
        self.assertEqual(snapshot.fairness_key, "pdf")
        self.assertEqual(snapshot.pending, 1)
        self.assertEqual(snapshot.active, 0)
        self.assertEqual(snapshot.rejected_per_key_limit, 1)

    def test_executor_queue_limit_has_stable_429_error(self) -> None:
        queue = self._queue(max_pending_per_key=4, max_pending_per_executor=1)
        self._enqueue(queue, key="a", text="first")

        with self.assertRaises(RequestAdmissionError) as exc_info:
            self._enqueue(queue, key="b", text="second")

        self.assertEqual(exc_info.exception.code, "executor_queue_full")
        self.assertEqual(exc_info.exception.status_code, 429)
        self.assertEqual(queue.pending_count, 1)
        self.assertEqual(queue.rejected_executor_limit, 1)

    def test_drain_removes_pending_work_from_every_key(self) -> None:
        queue = self._queue()
        self._enqueue(queue, key="a", text="a")
        self._enqueue(queue, key="b", text="b")

        drained = queue.drain(now=1.0)

        self.assertEqual({job.fairness_key for job in drained}, {"a", "b"})
        self.assertEqual(queue.pending_count, 0)
        self.assertEqual(queue.snapshots(now=1.0), ())


class LoadedModelExecutorFairnessTests(unittest.TestCase):
    def test_backend_failure_is_charged_and_releases_the_slot(self) -> None:
        def fail(request: ResponseRequest) -> EngineResult:
            del request
            time.sleep(0.01)
            raise RuntimeError("backend failed")

        executor = LoadedModelExecutor(
            model_name="test-model",
            replicas=[
                ReplicaRegistration(
                    replica_id="test-model#1",
                    complete_fn=fail,
                    runtime_capability=1,
                )
            ],
            configured_target_inflight=1,
            fairness_settings=FairnessSettings(),
        )
        executor.start()
        try:
            future = executor.enqueue(
                ResponseRequest(model="test-model", input="fail", fairness_key="pdf")
            )
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                future.result(timeout=1.0)

            snapshot = executor.snapshot()
            score = executor._pending_queue.score("pdf", now=time.perf_counter())

            self.assertEqual(snapshot.runtime_inflight, 0)
            self.assertIsNotNone(score)
            self.assertGreaterEqual(score, 5.0)
        finally:
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_queue_limit_applies_while_one_request_is_running(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def complete(request: ResponseRequest) -> EngineResult:
            entered.set()
            release.wait(timeout=1.0)
            return EngineResult(text=str(request.input))

        executor = LoadedModelExecutor(
            model_name="test-model",
            replicas=[
                ReplicaRegistration(
                    replica_id="test-model#1",
                    complete_fn=complete,
                    runtime_capability=1,
                )
            ],
            configured_target_inflight=1,
            fairness_settings=FairnessSettings(max_pending_per_key=1),
        )
        executor.start()
        try:
            first = executor.enqueue(
                ResponseRequest(model="test-model", input="first", fairness_key="pdf")
            )
            self.assertTrue(entered.wait(timeout=1.0))
            second = executor.enqueue(
                ResponseRequest(model="test-model", input="second", fairness_key="pdf")
            )

            with self.assertRaises(RequestAdmissionError) as exc_info:
                executor.enqueue(
                    ResponseRequest(model="test-model", input="third", fairness_key="pdf")
                )

            self.assertEqual(exc_info.exception.code, "fairness_key_queue_full")
            self.assertEqual(executor.snapshot().queue_depth, 1)
            release.set()
            self.assertEqual(first.result(timeout=1.0).text, "first")
            self.assertEqual(second.result(timeout=1.0).text, "second")
        finally:
            release.set()
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_capacity_four_gives_each_waiting_key_a_slot_before_borrowing(self) -> None:
        release = threading.Event()
        entered: list[str | None] = []
        entered_lock = threading.Lock()
        four_entered = threading.Event()

        def complete(request: ResponseRequest) -> EngineResult:
            with entered_lock:
                entered.append(request.fairness_key)
                if len(entered) == 4:
                    four_entered.set()
            release.wait(timeout=1.0)
            return EngineResult(text=str(request.input))

        executor = LoadedModelExecutor(
            model_name="test-model",
            replicas=[
                ReplicaRegistration(
                    replica_id="test-model#1",
                    complete_fn=complete,
                    runtime_capability=4,
                )
            ],
            configured_target_inflight=4,
            fairness_settings=FairnessSettings(soft_max_inflight_per_key=1),
        )
        futures: list[Future[EngineResult]] = []
        for index in range(4):
            futures.append(
                executor.enqueue(
                    ResponseRequest(
                        model="test-model",
                        input=f"a{index}",
                        fairness_key="a",
                    )
                )
            )
            futures.append(
                executor.enqueue(
                    ResponseRequest(
                        model="test-model",
                        input=f"b{index}",
                        fairness_key="b",
                    )
                )
            )
        executor.start()
        try:
            self.assertTrue(four_entered.wait(timeout=1.0))
            with entered_lock:
                first_four = list(entered)
            self.assertIn("a", first_four)
            self.assertIn("b", first_four)
            self.assertEqual(executor.snapshot().runtime_inflight, 4)
            self.assertEqual(executor.snapshot().queue_depth, 4)

            release.set()
            for future in futures:
                future.result(timeout=1.0)
        finally:
            release.set()
            executor.begin_shutdown()
            executor.join(timeout=1.0)

    def test_two_replicas_share_one_fairness_queue(self) -> None:
        release = threading.Event()
        entered: list[tuple[str, str | None]] = []
        entered_lock = threading.Lock()
        two_entered = threading.Event()

        def complete_for(replica_id: str):
            def complete(request: ResponseRequest) -> EngineResult:
                with entered_lock:
                    entered.append((replica_id, request.fairness_key))
                    if len(entered) == 2:
                        two_entered.set()
                release.wait(timeout=1.0)
                return EngineResult(text=replica_id)

            return complete

        executor = LoadedModelExecutor(
            model_name="test-model",
            replicas=[
                ReplicaRegistration(
                    replica_id=replica_id,
                    complete_fn=complete_for(replica_id),
                    runtime_capability=1,
                )
                for replica_id in ("test-model#1", "test-model#2")
            ],
            configured_target_inflight=1,
            fairness_settings=FairnessSettings(soft_max_inflight_per_key=1),
        )
        first = executor.enqueue(
            ResponseRequest(model="test-model", input="a", fairness_key="a")
        )
        second = executor.enqueue(
            ResponseRequest(model="test-model", input="b", fairness_key="b")
        )
        executor.start()
        try:
            self.assertTrue(two_entered.wait(timeout=1.0))
            with entered_lock:
                started = list(entered)
            self.assertEqual({replica_id for replica_id, _ in started}, {"test-model#1", "test-model#2"})
            self.assertEqual({key for _, key in started}, {"a", "b"})

            release.set()
            first.result(timeout=1.0)
            second.result(timeout=1.0)
        finally:
            release.set()
            executor.begin_shutdown()
            executor.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
