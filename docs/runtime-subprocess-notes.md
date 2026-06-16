# Runtime Subprocess Notes

This note captures the intended process boundary for runtime isolation in `llm-pool-dev`.

It is a design note, not an implementation spec.

It exists to preserve the architectural decisions behind a future subprocess-based runtime model, so those decisions do not need to be reconstructed later.

Current reality note:

- the general subprocess-isolated runtime architecture described here is not implemented
- `llm-pool` does now have `llama_server` and `vllm_serve` backends that start local server subprocesses for one loaded model runtime
- those subprocesses are backend-owned lifecycle plumbing, not a general parent/child runtime transport shared by all backends
- those backend-owned subprocesses already provide practical dependency isolation for their specific runtimes: separate binaries, environment variables, `LD_LIBRARY_PATH`, virtualenvs, CUDA/shared-library stacks, and even upstream forks can be selected per configured model
- model-specific native/runtime flags still live in backend config in this interim design; for example `vllm_serve_extra_args` can carry upstream `vllm serve` flags such as `--max-num-seqs`
- the first in-process scheduler/executor layer and runtime admin API already exist, so references below to an upcoming scheduler should be read as historical design context

## Purpose

The service already has:

- one API process
- one model router
- multiple backend-specific runtime implementations
- a first scheduler layer described in `runtime-scheduler-notes.md`

Today all loaded runtimes live in the same Python process as the API and router.

The goal of this note is to define when and why we may want to move loaded model runtimes into child processes, while keeping:

- scheduler policy in the parent process
- backend execution state in the runtime layer
- one common runtime contract across backends

## Main Architectural Decision

If runtime subprocess isolation is introduced, the intended boundary is:

- one child process per loaded model runtime

not:

- one child process per backend type
- one child process for the whole service
- a mixed model where some runtimes are local-only and others are process-isolated by default

In this note, "runtime" means:

- one loaded model instance
- with its own backend execution state
- owned by one lifecycle entry such as `loading`, `loaded`, `unloading`, `failed`

This matches the runtime state model already used by the admin/control plane.

## Why The Boundary Should Be Per Loaded Runtime

This keeps the system aligned with the runtime lifecycle rather than with backend names.

Benefits:

- failures are isolated to one loaded model runtime
- unload and drain behavior stays local to one runtime
- scheduler queues can target one runtime directly
- future capacity tuning stays expressed per model runtime
- the parent does not need special cases for backend grouping

This also fits the scheduler note's language around:

- scheduler-owned external queues
- runtime-owned execution state

See `runtime-scheduler-notes.md`.

## Why Subprocess Isolation Is Worth Considering

The main reasons are:

- fault isolation and lifecycle safety for native runtimes
- dependency, ABI, and shared-library isolation between runtime stacks

Examples of issues a subprocess boundary helps contain:

- backend crashes or segfaults
- native deadlocks or hangs
- unstable upstream backend upgrades in native inference stacks
- incomplete memory release during unload
- runtime-specific global state or allocator state that is safer to discard by process exit

It also allows different loaded runtimes to use different upstream stacks.

Examples:

- two versions of the same upstream project, such as separate llama.cpp or vLLM builds
- a pinned upstream release for one model and a newer upstream release for another
- a temporary fork for one backend without forcing the whole service onto that fork
- different CUDA, cuBLAS, cuDNN, NCCL, or other shared-library versions where the runtime can make that work safely
- different `LD_LIBRARY_PATH`, Python virtualenv, or native plugin search paths per runtime

In-process runtimes have to share one Python process, one imported module graph, one dynamic linker state, and one effective CUDA/shared-library environment.

That forces `llm-pool` toward the greatest common denominator of all loaded backend dependencies. A subprocess boundary lets each runtime carry the dependency stack it actually needs while the parent keeps the stable control-plane contract.

This matters more once the service also owns:

- its own scheduler
- its own queues
- runtime load and unload control
- runtime drain semantics

Without a process boundary, one native runtime failure can take down:

- the API
- the scheduler
- all queue state
- all other loaded runtimes

## Why A Uniform Boundary Is Preferable

If subprocess isolation is adopted for one backend, it is likely cleaner to use the same runtime boundary for all backends as well.

The reason is not that all backends are equally risky.

The reason is architectural consistency.

A mixed model would force the parent to reason about two runtime categories:

- runtimes that are direct in-process objects
- runtimes that are remote child-process handles

That tends to leak into:

- scheduler code
- load/unload code
- timeout handling
- drain behavior
- metrics
- tests

If the subprocess boundary exists at all, the default design should be:

- scheduler sees one runtime contract
- all backends implement that contract the same way from the parent's point of view

## Relation To The Scheduler Notes

This note is intentionally downstream of `runtime-scheduler-notes.md`, not a replacement for it.

The scheduler note defines the conceptual boundary:

- scheduler owns policy and external queues
- runtime owns backend execution state

This note defines where that runtime execution state may physically live:

- in a child process instead of in the parent process

The scheduler-facing runtime adapter from the scheduler note should remain valid.

The important shift is:

- in an in-process design, the adapter can wrap local runtime objects
- in a subprocess design, the adapter becomes the parent-side runtime handle that speaks IPC to the child

The scheduler should not care which transport is used.

## Parent And Child Responsibilities

### Parent Process Owns

- HTTP API
- request validation
- scheduler queues
- dequeue policy
- timeouts before runtime submission
- load and unload orchestration
- runtime registry and lifecycle state
- metrics aggregation
- admin/control API

### Child Runtime Process Owns

- backend imports
- model loading
- tokenizer and prompt preparation inside the runtime
- backend-specific execution objects
- backend-specific inflight job bookkeeping
- completion and failure production
- backend cleanup on shutdown

This is the same ownership split as the scheduler note, but with a real process boundary behind it.

## The Runtime Adapter Still Matters

The runtime adapter should remain the scheduler-facing contract.

It should not be thought of as "just the subprocess client".

It is the stable boundary between:

- scheduler policy
- runtime execution

If subprocesses are used, the adapter will internally communicate with the child process.

Conceptually:

- scheduler talks to adapter
- adapter talks to child runtime
- child runtime talks to backend implementation

That lets the scheduler stay transport-neutral.

## Recommended V1 Scope

The recommended first version is intentionally conservative:

- one child process per loaded runtime
- one active submitted job per runtime
- scheduler may still queue multiple jobs externally
- runtime-submitted work drains on unload
- no attempt yet to expose backend-specific multi-job concurrency

In other words:

- first make the boundary correct
- then make the scheduler correct
- only later increase per-runtime inflight concurrency

This keeps the first subprocess version small and architecturally honest.

## Why V1 Should Start With One Inflight Job Per Runtime

Even if some backends can handle multiple inflight jobs, that should not be the first concern.

The first concern is correctness of:

- runtime lifecycle
- queue ownership
- submission handoff
- completion handoff
- error propagation
- unload and drain behavior

Starting with one inflight job per runtime makes it easier to validate:

- scheduler queue semantics
- runtime state transitions
- child liveness detection
- result delivery
- parent/child recovery paths

Once the boundary is stable, the same runtime contract can be expanded to support:

- `target_inflight() > 1`
- backend-native batching
- backend-native multi-job execution

## Load And Unload Semantics

If subprocess runtimes are adopted, unload should stay aligned with the scheduler note.

Intended rule:

- queued work owned by the scheduler is cancelled
- work already submitted to the runtime is allowed to drain in v1
- runtime resources are released only after the drain completes

That means runtime unload should not default to:

- force kill active GPU work

However, the parent should still retain an operator-level recovery path for genuinely unhealthy children:

- crash detection
- timeout detection
- process death handling
- explicit restart or replacement

These are recovery semantics, not normal unload semantics.

## Failure Model

The parent should assume child processes can fail.

That should be treated as normal architecture, not as an exceptional edge case.

Examples:

- child exits during load
- child exits during inference
- child becomes unresponsive
- child returns a structured runtime error

The parent should translate these into stable runtime states such as:

- `failed`
- `unloaded`
- `loading`
- `loaded`
- `unloading`

and into scheduler-visible job outcomes such as:

- completed
- failed
- cancelled

## Transport Neutrality

This note does not prescribe a transport yet.

Possible transports include:

- stdin/stdout with structured messages
- Unix domain sockets
- loopback HTTP
- local IPC files plus a control channel

The important requirement is not the transport choice.

The important requirement is that the scheduler-facing adapter contract should not depend on transport details.

## Suggested Implementation Order

The preferred order is:

1. Implement the scheduler-facing runtime adapter boundary from `runtime-scheduler-notes.md`.
2. Keep the first adapter implementation local and simple if that helps development velocity.
3. Ensure the scheduler only talks to the adapter contract and never to backend internals.
4. Introduce child-process runtimes behind the same adapter contract.
5. Keep per-runtime inflight at `1` until the process and scheduler model are proven correct.
6. Only then consider backend-specific multi-job concurrency.

This order is acceptable because the adapter boundary is the real long-term abstraction.

If that boundary is kept clean, moving from local adapters to subprocess-backed adapters should be a transport swap rather than a scheduler rewrite.

## What The Scheduler Must Not Know

The scheduler should not know about:

- `ctranslate2.Generator`
- ExLlama `Job`
- ExLlama `Generator.iterate()`
- `llama_cpp.Llama`
- backend-specific locks
- backend-specific tokenizer object types
- child transport message shapes

Those belong inside the runtime implementation or the parent-side adapter.

## What This Note Intentionally Does Not Define

This note does not define:

- the exact IPC protocol
- the exact thread model inside the child
- token streaming across the process boundary
- active-job hard cancellation semantics
- persistent or disk-backed queues
- retry policy
- fairness policy across models
- detailed metrics schema

Those can be decided later.

## Practical Conclusion

If runtime subprocess isolation is introduced, the intended design is:

- keep API, scheduler, and control plane in the parent
- isolate loaded model runtimes as child processes
- use one child per loaded model runtime
- keep one common runtime adapter contract across all backends
- start with one inflight submitted job per runtime
- only add backend-specific concurrency after the boundary and scheduler are correct

That is the smallest design that stays aligned with both:

- the existing runtime lifecycle model
- the upcoming scheduler architecture
