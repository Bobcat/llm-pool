# Runtime Scheduler Notes

Companion working tracker: [runtime-scheduler-tracker.md](runtime-scheduler-tracker.md)

This note captures the scheduler design discussion around CT2, ExLlamaV3, and other `llm-pool` backends.

It is intentionally a design note, not an implementation spec.

Current reality note:

- `llm-pool` already has a first in-process scheduler/executor layer
- requests are already queue-backed per public model
- the runtime admin API already uses the scheduler boundary for load/unload semantics
- `llama_server` and `vllm_serve` now run through the same scheduler path, while their native subprocess lifecycles remain backend-owned
- scheduler-visible capacity for local backends is still conservative: `vllm_serve` may use vLLM's internal scheduler, prefix cache, CUDA graphs, and speculative decoding, but `llm-pool` still treats it as one submitted request per loaded runtime
- backend-native tuning such as `vllm_serve_extra_args: ["--max-num-seqs", "1"]` is model config, not scheduler policy
- this note now describes the broader scheduler design space beyond that first implemented cut

## Why This Is Worth Doing

The service already exposes one API contract for multiple backends, but the backend control surfaces are still very different:

- `Ct2ModelRuntime` mainly wraps a `ctranslate2.Generator` plus a tokenizer.
- `ExLlamaV3ModelRuntime` exposes more explicit runtime pieces such as `model`, `cache`, `generator`, `job_class`, and `sampler_class`.

That difference matters once we want our own queue and scheduler in front of inference.

An external queue is still useful even if a backend already has some internal queueing:

- we may want per-model admission control
- we may want predictable queue limits and backpressure
- we may want consistent timeout and cancellation behavior
- we may want scheduler-owned priorities or fairness
- we may want one place for queue metrics regardless of backend

The key design goal is to keep those policy decisions outside backend-specific code.

## Current Behavior

Today there is already a first scheduler layer in front of the engines, but not yet the fuller backend-native design described below.

### CT2

CT2 hides most of its runtime state inside `ctranslate2.Generator`. That is why `Ct2ModelRuntime` is small:

- `config`
- `generator`
- `tokenizer`
- `prompt_token_cache`

The tokenizer does the same conceptual job as in ExLlamaV3: turn text into model input and generated tokens back into text. The API shape is different, but the role is the same.

The `prompt_token_cache` is only a small CPU-side optimization for repeated static prompts. It matters more for long repeated instructions than for short prompts.

### ExLlamaV3

ExLlamaV3 exposes more of the mechanics directly, so the runtime keeps more members:

- `config`
- `model`
- `cache`
- `tokenizer`
- `generator`
- `job_class`
- `sampler_class`
- `generation_lock`

That does not mean all members are used in the hot path on every request. It means the backend API makes those concepts explicit, so the runtime holds them explicitly too.

One important detail:

- the current `ExLlamaV3Engine.complete()` path serializes requests per runtime with `generation_lock`
- ExLlamaV3 itself can work with multiple jobs in one generator
- the current code does not exploit that batching behavior yet because one request holds the runtime until its own job completes

So the future opportunity is not "load the same model multiple times", but "keep one loaded model and let one runtime manage multiple in-flight jobs".

## Why Option 2 Is The Cleaner Path

If we build our own queue and scheduler, there are two broad options:

1. The scheduler knows CT2 details and ExLlamaV3 details directly.
2. The scheduler talks to a small runtime adapter interface and each backend implements that interface in its own way.

Option 2 is the cleaner path.

Why:

- the scheduler should own policy, not backend mechanics
- CT2 and ExLlamaV3 differ a lot internally, and that difference should stay localized
- we can start simple per backend and still keep one scheduler shape
- future backends can plug into the same contract

The scheduler should not need to know about:

- `generate_batch`
- ExLlamaV3 `Job`
- ExLlamaV3 `ComboSampler`
- ExLlamaV3 `generation_lock`
- backend-specific token shapes

Those belong inside the runtime adapter.

## Proposed Runtime Adapter

A useful interface is one that lets the scheduler:

- know how much work a runtime should hold
- hand work to the runtime
- advance the runtime if the backend needs active driving
- receive completions or failures back

The important design choice is that this should be a `tick()`-style interface, not just a passive `poll()`.

Reason:

- ExLlamaV3 needs active progress through `generator.iterate()`
- CT2 can probably be implemented behind worker threads or futures and just report finished work on `tick()`

So one common interface can still fit both.

Example shape:

```python
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class RuntimeEvent:
    job_id: str
    kind: Literal["started", "completed", "failed", "cancelled"]
    text: str | None = None
    error: str | None = None
    metrics: object | None = None


class SchedulableModelRuntime(Protocol):
    model_name: str

    def target_inflight(self) -> int:
        """Scheduler hint for how many jobs this runtime should hold."""

    def inflight(self) -> int:
        """Jobs already handed to the backend and not finished yet."""

    def submit(self, job_id: str, request: object) -> None:
        """Accept one queued request from the scheduler."""

    def tick(self) -> list[RuntimeEvent]:
        """Advance backend work and emit new events."""

    def cancel(self, job_id: str) -> bool:
        """Best-effort cancellation for a queued or active job."""
```

This is intentionally small.

It gives the scheduler a stable contract without baking backend-specific assumptions into scheduler code.

## Why These Methods

`target_inflight()`

- gives the scheduler a backend-owned capacity hint
- for ExLlamaV3 this would likely be near `exllama_max_batch_size`
- for CT2 this could start at `1` and later become a configured parallelism hint

`inflight()`

- tells the scheduler how many jobs are already handed off
- this is the clean boundary between external queue policy and backend execution state

`submit(...)`

- keeps backend job creation inside the runtime adapter
- ExLlamaV3 can create sampler/job objects internally
- CT2 can prepare the prompt and dispatch a worker or direct call internally

`tick()`

- lets ExLlamaV3 drive `generator.iterate()`
- lets CT2 collect finished futures or worker results
- gives one common heartbeat to the scheduler

`cancel(...)`

- keeps cancellation behavior runtime-specific
- makes it possible to support external queue timeouts later without leaking backend details

## Ownership Boundary

The clean split should be:

### Scheduler owns

- external pending queues
- dequeue policy
- fairness and priority
- timeouts before submission
- global metrics such as queue wait time

### Runtime adapter owns

- tokenizer use
- backend-specific request preparation
- backend-specific execution objects
- progress/completion mapping
- backend cancellation mechanics

That split keeps the architecture honest.

It also avoids a design where the scheduler slowly accumulates one `if backend == ...` branch after another.

## How This Would Map To ExLlamaV3

For ExLlamaV3 the adapter would likely:

- keep one loaded `model`
- keep one `cache`
- keep one `generator`
- keep local bookkeeping from our `job_id` to ExLlamaV3 job object
- use `tick()` to call `generator.iterate()`
- translate ExLlamaV3 events into `RuntimeEvent`

This is the important part:

- one loaded model
- one runtime
- multiple in-flight jobs inside that runtime

That is the path that preserves shared VRAM allocations for model weights while still allowing multiple active requests.

Creating multiple runtimes by reloading the same model would move in the wrong direction because it risks duplicating VRAM usage.

## How This Would Map To CT2

For CT2 the adapter would likely stay much thinner.

Possible shape:

- keep one loaded `Generator`
- keep one tokenizer
- keep prompt preprocessing inside the adapter
- submit work by starting a backend call or dispatching a worker future
- report completion on `tick()`

The important point is not that CT2 and ExLlamaV3 behave identically internally. They do not.

The important point is that they can still present the same scheduler-facing control surface.

## A Reasonable Scheduler Loop

At a high level, the scheduler loop could look like this:

```python
for runtime in runtimes:
    events = runtime.tick()
    handle_runtime_events(events)

    while runtime.inflight() < runtime.target_inflight():
        queued = queue.pop(runtime.model_name)
        if queued is None:
            break
        runtime.submit(queued.job_id, queued.request)
```

That loop is deliberately boring.

That is a good sign.

It means the scheduler only decides:

- which model queue to pull from
- how much work to hand off
- what to do with completions, failures, and timeouts

It does not need to know how CT2 or ExLlamaV3 actually execute.

## Concurrency Sweet Spot

For a high-end single-GPU workstation setup, the sweet spot is usually not "as many requests as fit in VRAM".

In practice the useful concurrency is often limited earlier by:

- memory bandwidth
- KV-cache pressure
- batching overhead
- scheduler overhead
- latency targets such as TTFT and p95 response time

So the right unit to think in is usually:

- active sequences per loaded model

not:

- number of models that can theoretically be loaded
- number of requests that can be admitted before OOM

For a 96 GB Blackwell-class prosumer GPU, a reasonable starting heuristic is:

- small quantized models around `2B-4B`: often `8-16` in-flight requests on one loaded model
- mid-sized quantized models around `7B-14B`: often `4-8`
- larger quantized models around `27B-32B`: often `2-4`
- very large models that already consume a large share of VRAM: often `1-2`, sometimes `3-4` if throughput matters more than latency

These are not guarantees. They are only starting points.

The actual sweet spot depends heavily on:

- prompt length
- output length
- quantization format
- cache size and cache quantization
- batching behavior of the backend
- whether we optimize for throughput or latency

The important takeaway is:

- more concurrency is not automatically more throughput
- after a certain point, latency often degrades faster than total tokens/sec improves

That matters for both a future external scheduler and backend-native batching.

## How To Tune It In Practice

If we ever want to tune this properly, the simplest useful approach is a short concurrency sweep per model:

- test `1, 2, 4, 8, 12, 16, 24`
- keep prompt length and output length fixed for the benchmark
- record TTFT, total latency, output tokens/sec, and GPU memory usage

Then stop increasing concurrency once one of these happens:

- total throughput improves only marginally from one step to the next
- TTFT or p95 latency starts degrading sharply
- cache pressure makes runtime behavior unstable

That gives a practical operating point instead of a theoretical maximum.

## Interaction With Runtime Unload

The runtime admin API now exists and should stay aligned with the scheduler boundary described above.

The clean split is:

- scheduler owns external pending queues
- runtime owns backend execution state

That means unload should behave like this:

- new requests for the model are rejected once unload starts
- requests still waiting in a scheduler-owned queue should be cancelled
- requests already submitted to the runtime should be allowed to drain in v1
- only after that drain completes should the runtime be released

So "unload" should not mean "immediately kill whatever the backend is doing".

The implemented scheduler-aware rule is:

- queued work: cancel
- runtime-submitted work: drain

That keeps cancellation policy in the scheduler layer and avoids pretending that all backends can safely hard-cancel active GPU work.

## Out Of Scope For Now

This note does not define:

- exact thread model
- exact active-job cancellation semantics inside each backend
- streaming token delivery contract
- persistence or disk-backed queues
- retry policy
- fair scheduling algorithm across models

Those can come later.

For now the main architectural idea is:

- external queue and scheduler are still useful
- ExLlamaV3 should use one loaded model with multiple in-flight jobs, not multiple reloaded runtimes
- the scheduler should talk to a small runtime adapter interface, not to backend internals
Note: public-model replica routing is described separately in `model-replica-routing-notes.md`. The scheduler note here still applies at the per-loaded-runtime boundary.
