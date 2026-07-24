# Client Fairness Notes

Companion to [runtime-scheduler-notes.md](runtime-scheduler-notes.md). That note
lists "fairness and priority" as scheduler-owned but defers the algorithm. This
note proposes the algorithm.

It is a design note, not an implementation spec.

## Scope

This is about fairness **between clients inside one model executor's pending
queue**.

Each loaded model has its own executor, its own pending queue, and its own
`target_inflight` slots (see the scheduler note). Several clients can send to the
same model. They all land in that one executor's queue and compete for the same
slots. This note is about how that queue picks who goes next.

It is **not** about fairness across models. Arbitrating GPU time between different
model executors is a separate, still-deferred concern.

## The problem

The executor's pending queue is a flat FIFO. `_pending_jobs` is a `deque`, and
`_run_loop` takes the head with `popleft` (`app/engine/scheduler.py`). The queue
has no notion of which client a job came from.

So a client that submits many calls at once takes that many consecutive places at
the front. Every other client waits behind all of them.

This is a real workload, not a corner case:

- the document pipeline translates a PDF page by page and fans its pages out
  concurrently, so it holds many calls to the large model outstanding at once.
- an interactive client — a prompt-testing workbench — sends one call to the same
  large model and wants it back quickly.

Under the FIFO, the interactive call waits behind the whole page fan-out. Both
clients carry the same model id, so they share one executor and one queue.

## Why this belongs in the pool

A client can limit its own concurrency, and the document pipeline does. But that
is cooperative, not enforced. It breaks the moment a client misbehaves:

- a bug — a retry storm, or a client that ignores its own limit
- a new client that was never taught the convention

Fairness has to be enforced where the scarce resource is. That is the pool.
Politeness on the client side is a best effort on top, not the guarantee.

## The algorithm: weighted fair queuing on served time

Track, per client, the **cumulative time the pool has spent serving it**. When a
slot frees, admit the next queued call from the client with the lowest served
time. Weight it per client, so the comparison is `served_time / weight`.

Round-robin over clients is not enough. Round-robin equalises **turns**, not
time. If one client's calls run 60s and another's run 2s, equal turns is maximally
unequal time. LLM generations hold their slot for the whole generation, so time is
what has to be equalised.

Concretely:

- **counter** — one `served_time` per fairness key.
- **admit** — on a free slot, pick the queued call whose key has the lowest
  `served_time / weight`.
- **charge on completion** — when a call finishes, add its measured duration to
  that key's counter. The pool already measures this: `engine_total_wall_ms` in
  `ResponseMetrics`. Admit on the counter, charge when done.
- **weight** — a per-key multiplier. A higher weight buys more served time. This
  is how an interactive client outranks a batch client, with one knob instead of
  separate priority queues.

The scheme is work-conserving. If only one client is active, it has the lowest
counter by default and gets every slot. Fairness only spreads load when more than
one client is actually waiting.

### Decay, not a lifetime sum

A plain lifetime total starves a returning client. A client that was heavy an hour
ago and comes back idle carries a huge counter, and gets nothing until the others
catch up. A brand-new client at zero gets the opposite — a windfall that lets it
grab every slot until it catches up. This is the newly-active-flow problem from
classic weighted fair queuing.

Two fixes, either works:

- **decay the counters** — an exponential moving average, so the counter reflects
  a recent window instead of all history. This is what "equal time on average"
  should mean.
- **clamp on re-activation** — when an idle client becomes active again, raise its
  counter to at least the current minimum. No windfall, no penalty.

The decay is usually the nicer of the two. Old imbalance fades on its own.

### Long generations still need a per-client slot cap

Weighted fair queuing fixes admission **order**. It does not fix slot **holding**.

A client whose calls are admitted first, and then each run 60s, holds those slots
for 60s. A second client waits for a slot to free, even under fair ordering. Order
fairness is not throughput fairness when a single call occupies a slot for a long
time.

The complement is a **soft per-client inflight cap**: a client may hold at most N
of the executor's slots at once, but only while another client has work queued.
Alone, it may use all slots. This bounds how long one client's long calls can
delay another. It is work-conserving for the same reason the queue is.

This is the piece the asr-pool did not need. Its jobs release their slots quickly,
so admission-order fairness is close enough to throughput fairness there.

## Identity

The fairness key is a self-declared string on the request. Add it as an optional
field on `ResponseRequest`, alongside the recently added `prompt_cache_key`.
Default it to one shared "unknown" key.

No authentication layer. The pool's clients are our own services, and the threat
is a client bug, not an adversary. A buggy client reuses its identity; it does not
mint fresh ones to cheat. Keying on the service name — `document-pipeline`,
`workbench`, `realtime` — is enough, and it makes a storm from one buggy client
both attributable and self-throttling.

A self-declared key is spoofable. A client that mints a fresh key per request
escapes its own accumulated cost — the Sybil attack on fair scheduling. No
scheduling formula defends against it; only unforgeable identity does. That is out
of scope while the fleet is trusted. If untrusted clients ever arrive, put a
gateway service in front that authenticates and assigns the key. The pool keeps
trusting the key it receives, and the trust boundary moves to the gateway.

## Migration: allow requests without a key

Not every client will send a key at once. So allow both, in two tiers:

- **with a key** — fair-queued among themselves, as above.
- **without a key** — served when a slot is free and no keyed request is queued.
  Among themselves, plain FIFO: first come, first served.

Unkeyed requests keep working, they just fall to the back. This lets clients adopt
the key one at a time. Sending a key earns a fair share; not sending one earns the
leftovers.

Strict "always after" starves the unkeyed tier under continuous keyed load. Guard
it with an anti-starvation cap, the same shape asr-pool uses
(`interactive_burst_max`): after N consecutive keyed admissions, let one unkeyed
request through. "After" becomes "strongly deprioritised", not "never".

Sending no key puts a client at the back, so no one games their way into the
unkeyed tier. The only cheat left is minting keys in the keyed tier, which is the
trusted-fleet question above.

## Defense in depth: a per-key cap

Fairness shares time. It does not bound absolute damage. Add a per-key queue-depth
or rate cap as a floor under it, so one key — a bug, a runaway loop — cannot grow
the queue without limit. This cap needs no identity guarantee to be useful.

## Relationship to asr-pool

asr-pool already has this model, in its scheduler:

- reserved slots for a priority class
- round-robin over `fairness_key` within that class
- an `interactive_burst_max` anti-starvation cap

The llm-pool work is one upgrade on it: **round-robin over turns becomes weighted
fair queuing over served time**, plus the soft per-client inflight cap. Both
follow from the same difference — asr jobs release their slots quickly, llm
generations hold them.

The class split (interactive vs normal) is not needed here. Different workloads
already use different model ids, so they land on different executors and never
share a queue. The per-key weight covers priority within a shared model.

## Where it hooks in

All per-executor. Each executor keeps its own fair queue; nothing global changes.

- `ResponseRequest` — add the optional `fairness_key` field.
- `enqueue` — bucket the job by fairness key instead of appending to one flat
  deque.
- the `_run_loop` selection — replace `popleft` with "pick from the key with the
  lowest `served_time / weight`, respecting the soft inflight cap and the
  keyed-before-unkeyed tiering".
- completion handling — add the finished call's `engine_total_wall_ms` to its
  key's counter, and decay counters over time.

## Out of scope

- fairness across models — arbitrating GPU time between different model executors
- authentication and token issuance — deferred to a future gateway
- the exact decay constant and per-client cap value — tune against real load, the
  way the scheduler note tunes `target_inflight`
