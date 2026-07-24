# Runtime Scheduler Tracker

This document is the working companion to [runtime-scheduler-notes.md](runtime-scheduler-notes.md).

The design note should stay architecture-focused.
This tracker is where we record implementation scope, accepted MVP cuts, work phases, and addenda as the scheduler work evolves.

Current reality note:

- the first in-process scheduler/executor layer is implemented
- public-model replica routing is implemented at aggregate admin/API level
- per-model weighted client fairness is implemented with per-key FIFO buckets,
  measured slot-time, and a work-conserving soft inflight cap
- per-key and per-executor pending limits provide bounded backpressure
- local runtime capability is still effectively `1` in-flight request per replica except for `openai_remote` and `vllm_serve`, which can use configured `target_inflight`
- `llama_server` and `vllm_serve` use the same scheduler/executor path; their native subprocess lifecycles are backend-owned, not the general runtime-subprocess design from `runtime-subprocess-notes.md`
- `vllm_serve` high-throughput tuning is currently expressed through model config and upstream flags, not through scheduler-native adaptive concurrency

## Status Labels

- `proposed`
- `in_progress`
- `done`
- `deferred`

## Original MVP Boundary

Status: `done`

This section records the first scheduler cut. Client fairness and bounded
backpressure were added later.

Included in the first scheduler MVP:

- a scheduler-owned pending FIFO queue per loaded model runtime
- a scheduler-facing runtime adapter per loaded model runtime
- configurable target inflight per model runtime
- default target inflight of `1`
- synchronous HTTP behavior for `POST /v1/responses`: enqueue, wait, then return the completed result
- unload integration with this rule:
  queued work is cancelled
  runtime-submitted work is drained

Explicitly out of scope for the first scheduler MVP:

- fair scheduling across models
- backpressure
- queue limits
- retry policy
- persistence or disk-backed queues
- hard cancellation of active backend work
- true token-by-token streaming from the scheduler/runtime boundary
- backend-specific adaptive concurrency heuristics

## Current Decisions

Status: `done`

- scheduler capacity is tracked per loaded model runtime, not per backend family
- the primary execution owner in `llm-pool` is a loaded model executor
- a loaded model executor is the `llm-pool` analogue of an `asr-pool` warm runner slot
- `target_inflight` is the scheduler-facing capacity knob
- the initial default for `target_inflight` is `1`
- `runtime_inflight` should mean jobs that are past the scheduler-owned cancel boundary
- if a backend cannot honestly support `target_inflight > 1` yet, the implementation should clamp rather than pretend
- ExLlamaV3 should move toward one loaded runtime with multiple in-flight jobs, not multiple runtimes for the same model
- the first scheduler implementation should live in a dedicated helper owned by `ModelRouterEngine`, not in FastAPI request handlers
- the first scheduler implementation should use the current synchronous request API:
  enqueue in the engine
  wait for one job result
  then return the completed response

## Chosen Model

Status: `done`

The chosen high-level concurrency model should be intentionally similar to `asr-pool`, but with a different execution owner.

Shared shape with `asr-pool`:

- a thin intake layer
- a scheduler-owned queue boundary
- long-lived execution owners behind that boundary
- explicit queued-versus-submitted lifecycle semantics
- scheduler/service-owned background loops

Important difference from `asr-pool`:

- `asr-pool` schedules onto fungible runner slots
- `llm-pool` schedules onto model-bound executors

That means the main scheduler unit in `llm-pool` is:

- one loaded model executor

not:

- one global slot
- one backend family

In the current in-process implementation, a model executor is a parent-side object that owns:

- one or more identical loaded runtime replicas
- one pending fairness queue with per-key FIFO buckets
- one driver loop
- one configured scheduler capacity
- one effective scheduler capacity

If subprocess isolation is introduced later, the same model executor concept should remain valid.
Only the implementation behind the executor changes:

- today: in-process runtime object
- later: parent-side handle to one runtime subprocess

## Concrete MVP Design

Status: `done`

### 1. Scheduler Ownership

- `ModelRouterEngine` remains the top-level owner of model registry, runtime state, and load/unload semantics
- a dedicated scheduler helper should be introduced under `ModelRouterEngine`
- FastAPI should not own queueing or scheduling policy directly

### 2. Executor Registry

For each loaded model, the engine should maintain one executor entry with:

- `model_name`
- `resolved_backend`
- `pending_queue`
- `configured_target_inflight`
- `effective_target_inflight`
- `runtime_inflight`
- `accepting_new_requests`
- `driver_thread` or equivalent background loop handle

### 3. Request Flow

`POST /v1/responses` should do this:

- validate the model and runtime state through the engine
- create one scheduler job with a result future
- enqueue the job onto that model executor
- wait for the future result
- return the existing JSON or SSE envelope shape

This keeps the public inference API synchronous even though execution is now queue-backed.

### 4. Driver Loop Shape

Each model executor should have a long-lived driver loop.

Its responsibilities are:

- collect completions or failures from already submitted runtime work
- reduce `runtime_inflight` when work finishes
- submit more queued jobs while `runtime_inflight < effective_target_inflight`
- sleep or wait on a condition when there is no new work

This is deliberately closer to the long-lived worker-loop model used in `asr-pool` than to one single global tick loop.

### 5. Runtime Adapter Contract

The scheduler-facing runtime adapter should stay small.

For the first patch, the implementation can be minimal:

- `submit(job)` or equivalent
- completion/failure handoff back to the driver loop
- an honest runtime capacity report used to derive `effective_target_inflight`

The design note's fuller `tick()` contract remains valid as a target shape, but the first implementation does not need to force every backend into the exact same polling mechanism on day one.

### 6. Capacity Semantics

- config should express `configured_target_inflight`
- runtime capability should express the current truthful maximum
- the executor should expose:
  `effective_target_inflight = min(configured_target_inflight, runtime_capability)`

This preserves the future-facing config knob without lying about actual backend concurrency.

### 7. Unload Semantics

When unload starts for one model executor:

- mark the executor as not accepting new requests
- reject new requests for that model
- cancel scheduler-owned queued jobs for that model
- let already submitted runtime work drain
- only then clean up runtime resources and remove the executor

### 8. Original Implementation Bias

The first implementation deliberately biased toward correctness over backend
ambition:

- preserve current API behavior
- preserve current model state semantics
- keep queue semantics explicit
- do not add fairness, priorities, queue limits, or global cross-model arbitration yet
- keep the first execution loops in-process

## Work Phases

### Phase 1: Scheduler Skeleton

Status: `done`

Goal:
- introduce the minimal queue and scheduler structure without changing the external inference API

Planned work:
- add scheduler job and result types
- add scheduler-owned pending queues keyed by model name
- add a scheduler-facing runtime adapter shape for currently loaded runtimes
- add a simple scheduler loop that advances runtimes and submits queued work up to `target_inflight`
- route `POST /v1/responses` through the scheduler path instead of calling the backend engine directly

Definition of done:
- requests for loaded models are enqueued and completed through the scheduler
- per-model runtime inflight counts are tracked by the scheduler
- the default behavior remains effectively serial when `target_inflight = 1`

### Phase 2: Configurable Capacity

Status: `done`

Goal:
- make per-model runtime concurrency configurable while keeping the MVP semantics small

Planned work:
- add config support for `target_inflight` with default `1`
- surface the effective value in admin output
- validate values conservatively

Definition of done:
- each configured model can set its own scheduler target inflight
- omitted values fall back to `1`

### Phase 3: Unload Integration

Status: `done`

Goal:
- align runtime unload with scheduler-owned queues

Planned work:
- reject new requests once unload starts
- cancel queued work for the unloading model
- allow already submitted runtime work to drain
- only release runtime resources after drain completes

Definition of done:
- unload semantics match the scheduler note for queued-vs-submitted work
- no queued work survives a completed unload for that model

## Open Questions

Status: `in_progress`

- whether a future backend-native streaming path should replace the current SSE behavior for `stream: true`
- whether replica routing should later move from least-inflight to normalized occupancy when replicas can expose different effective capacities

Questions now considered resolved for MVP:

- the scheduler should live under `ModelRouterEngine` as a dedicated helper, not in FastAPI handlers
- `target_inflight > 1` may be configured immediately, but the executor must clamp to truthful runtime capability when needed

## Addenda

### 2026-04-26: Initial MVP Cut

Status: `done`

- build the scheduler around per-model runtime ownership, not per-backend shared ownership
- keep the first scheduler implementation in-process
- prefer a small honest scheduler with `target_inflight = 1` by default over a larger design with premature fairness or backpressure features

### 2026-04-26: ASR Pool Comparison And Chosen Direction

Status: `done`

- `asr-pool` and `llm-pool` should share the same high-level structure:
  intake layer
  scheduler-owned queue boundary
  long-lived execution owners
  explicit queued-versus-submitted semantics
- the right `llm-pool` analogue of an ASR runner slot is a loaded model executor
- unlike ASR slots, model executors are model-bound rather than fungible
- the first `llm-pool` scheduler should therefore be organized as a registry of model executors with per-executor queues and capacities
- subprocess isolation later should preserve this shape rather than replace it

### 2026-04-27: Scheduler MVP Implemented

Status: `done`

- `POST /v1/responses` now runs through the in-process scheduler/executor layer
- queue ownership is per public model
- aggregate queue metrics are surfaced in admin responses:
  `queue_depth`
  `runtime_inflight`
  `configured_target_inflight`
  `effective_target_inflight`
- unload semantics now match the MVP rule:
  queued work is cancelled
  runtime-submitted work is drained

### 2026-04-27: Public-Model Replicas Implemented

Status: `done`

- one public model may load multiple identical replicas
- the scheduler dispatches across replicas of the same public model
- public model load/unload is aggregate
- replica selection uses least-inflight with round-robin tie-breaking
- admin remains aggregate per public model rather than per replica

### 2026-07-25: Per-Model Client Fairness Implemented

Status: `done`

- `ResponseRequest.fairness_key` identifies a stable scheduling identity
- omitted keys share one anonymous bucket
- every public model executor owns per-key FIFO buckets
- weighted least-served selection charges measured backend slot-time
- active elapsed time affects selection before completion
- a soft per-key inflight cap prevents monopoly while allowing unused capacity
  to be borrowed
- per-key and per-executor pending limits reject before enqueue with distinct
  HTTP `429` codes
- fairness state and rejection counters are visible through the bounded admin
  snapshot
- replica completion reports slot-time to the shared public-model fairness
  queue
- unload drains all pending key buckets and lets active work finish
