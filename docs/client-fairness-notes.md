# Client Fairness Notes

Companion to [runtime-scheduler-notes.md](runtime-scheduler-notes.md). That note
assigns fairness and priority to the scheduler but defers the policy. This note
defines the first per-client policy.

Status: implemented in the per-model scheduler. This note records the V1 policy,
its tradeoffs, and its test contract.

## Scope

This design covers fairness between scheduling identities inside one loaded
model's executor.

Each loaded model has its own executor, pending work, replicas, and runtime
slots. Calls to different model ids do not share an executor. This policy
therefore acts independently for every model.

V1 identifies work by a stable scheduling identity such as:

- `translation-service:image`
- `translation-service:pdf`
- `workbench`

It does not provide fairness between users, document runs, or sessions behind
one service. That would require trusted sub-client identity or hierarchical
fairness.

A caller may present more than one stable scheduling identity. For example,
translation-services uses the coarse task classes
`translation-service:image` and `translation-service:pdf`. The pool schedules
between keys, not originating services. A caller with more active keys therefore
gets a larger aggregate share under contention. V1 accepts this flat-key
tradeoff and does not attribute keys back to a service.

Fairness across model executors is out of scope. Different executors can still
compete for the same GPU outside this policy.

## Original problem

Before V1, `LoadedModelExecutor` stored pending jobs in one `deque` and
`_run_loop` always took its head with `popleft`.

A service that submits a large batch occupies consecutive queue positions. A
later interactive call to the same model waits behind that batch, even if it is
the only call from its service.

Client-side concurrency limits reduce this problem but cannot enforce fairness.
A retry storm, bug, or new client can ignore those limits. The pool owns the
scarce runtime slots, so it must enforce their allocation.

## Goals

The policy must:

- preserve FIFO order within one client key;
- share runtime slot-time between continuously backlogged keys according to
  pool-owned weights;
- stop one key from refilling every slot while another key is waiting;
- keep every usable slot occupied when work is queued;
- accept old requests that do not send a key;
- charge failed active calls for the time for which they occupied a slot;
- keep queue and shutdown behavior observable and bounded.

The policy is non-preemptive. It cannot reclaim a slot from an active call.

## Request identity

`ResponseRequest` accepts this optional field:

```python
fairness_key: str | None = None
```

A supplied key must:

- contain non-whitespace text;
- be trimmed before use;
- contain at most 128 characters after trimming.

An omitted key maps to one internal anonymous key. Anonymous work joins the same
fair scheduler with the default weight. It does not enter a lower-priority
migration tier.

This gives old clients normal service during rollout. Their calls share one
anonymous FIFO bucket until they adopt stable keys.

### Trust boundary

The request declares its own key. This is acceptable while all callers are
trusted services.

A caller can evade fairness by minting fresh keys. If untrusted callers are
introduced, an authenticated gateway must assign the key. The pool can keep the
same scheduling contract behind that gateway.

## Pool-owned weights

Requests declare identity, not scheduling power. A request must not carry its
own weight.

V1 uses one pool-wide mapping under `engine.fairness`:

```json
{
  "engine": {
    "fairness": {
      "default_weight": 1.0,
      "weights": {
        "translation-service:image": 1.0,
        "translation-service:pdf": 1.0,
        "workbench": 1.0
      },
      "soft_max_inflight_per_key": 1,
      "max_pending_per_key": 32,
      "max_pending_per_executor": 128,
      "idle_state_ttl_s": 300
    }
  }
}
```

Every executor reads the same mapping independently. A missing key uses
`default_weight`. Weights must be finite and greater than zero.

The same section owns these scheduler settings:

- `soft_max_inflight_per_key`;
- `max_pending_per_key`;
- `max_pending_per_executor`;
- `idle_state_ttl_s`.

A continuously backlogged key with weight `2` should receive about twice the
slot-time of a continuously backlogged key with weight `1`. Weights influence
sustained contention. They do not reserve a slot or preempt active work.

Per-model weight overrides are outside V1.

## Scheduling model

The policy is weighted least-served scheduling over runtime slot-time. It is not
classical weighted fair queuing: an LLM call's service cost is unknown until the
call releases its slot.

### Per-key state

Each executor keeps:

- one FIFO `deque` per normalized fairness key;
- `virtual_service` per key;
- active job start times per key;
- active job count per key;
- deterministic round-robin state for equal scores.

`virtual_service` stores completed slot-time after weight normalization.

### Service unit

Charge the elapsed time for which a job occupies a runtime slot:

```text
service_ms = backend_finished_at - backend_started_at
charge = service_ms / weight
```

The scheduler measures these timestamps around `_complete_fn` and charges from
that stopwatch directly.

Do not charge `engine_total_wall_ms`. That value is created after the scheduler
future completes and includes queue wait. Charging it would penalize a client
for time during which it received no service.

Charge elapsed slot-time in `finally`, including backend failures. A job
cancelled before runtime submission consumed no slot-time and receives no
charge.

With four concurrent slots, total charged service can grow by four milliseconds
per millisecond of wall time. That is intentional: the unit is sequence-slot
time, not exclusive GPU wall time.

### Selection score

Completed service alone lags while jobs are active. Selection must include
their current elapsed slot-time:

```text
score(key, now) =
    virtual_service[key]
    + sum(now - active_job.started_at for active jobs of key) / weight[key]
```

When a slot is free, choose a queued key with the lowest score. Take the oldest
job from that key's FIFO bucket.

Use round-robin order to break equal scores. Equal scores are common when an
executor fills several slots before any job has completed.

On completion:

1. remove the job from the key's active set;
2. add its measured `service_ms / weight` to `virtual_service`;
3. notify the executor loop that a slot and updated score are available.

Active counts and times cover all replicas belonging to the executor. Fairness
is per public model, not per replica.

### New and returning keys

A new key must not start at zero while existing contenders carry positive
service. Initialize it to the minimum current score among the other queued or
active keys. Use zero only when no other key is active or queued.

Keep idle key state for a bounded grace period. This prevents a bursty client
from resetting its history between consecutive calls. After the grace period,
discard the state. A returning key then re-enters at the current minimum.

V1 does not use exponential decay. The activation baseline prevents a new-key
windfall, while idle-state expiry removes old history. The checked-in grace
period is 300 seconds and remains configurable.

The implementation does not currently rebase counters. A future rebase may
subtract a common minimum without changing selection order.

## Soft per-key inflight cap

The score controls admission order. A soft inflight cap prevents one key from
holding every slot when another key is already waiting.

Let `N` be the configured soft maximum active jobs per key, with a minimum of
one. Selection works in two passes:

1. consider queued keys whose active count is below `N`;
2. if that set is empty while a slot is free, consider every queued key.

The second pass is borrowing. It keeps the executor work-conserving. The cap is
an anti-monopoly preference, not a hard quota or reserved capacity.

Example with four slots and `N = 1`:

- one active key may borrow all four slots;
- when a second key queues while the first holds three slots, the second key is
  the only below-cap candidate for the remaining slot;
- when both keys reach the soft cap and slots remain, the score decides who
  borrows them.

The cap applies across all replicas of the public model.

The cap cannot help until a slot becomes free. If one client already holds every
slot, a newly arrived client waits for the next completion. Avoiding that delay
would require idle reservation or preemption, neither of which is part of this
design.

## Admission procedure

For each free runtime slot:

1. collect keys with non-empty pending buckets;
2. apply the first soft-cap pass;
3. fall back to borrowing if the first pass has no candidates;
4. compute each candidate's score including active elapsed time;
5. choose the lowest score, using round-robin for ties;
6. pop the oldest job from that key;
7. record its active start time;
8. submit it to the selected replica.

Replica selection remains separate. The executor first chooses the fair job and
then uses the existing least-inflight and round-robin replica policy.

## Queue limits and backpressure

Fair scheduling does not bound memory or absolute damage from a broken client.
V1 needs both:

- a maximum pending depth per key;
- a maximum total pending depth per executor.

The total limit still protects the pool if a caller generates fresh keys.

Reject before enqueue when either limit is reached. The per-key limit returns
HTTP `429` with code `fairness_key_queue_full`. The total executor limit returns
HTTP `429` with code `executor_queue_full`. Rate limiting is a separate policy
and is outside V1.

## Shutdown and lifecycle

The key buckets preserve the existing model lifecycle behavior.

On unload:

- stop accepting new jobs;
- drain every pending key bucket;
- fail every drained future with `model_unloading`;
- let active jobs release normally;
- remove idle fairness state with the executor.

`queue_depth` remains the sum of every pending bucket. Idle state expires after
the configured TTL and disappears with the executor on unload.

## Observability

Aggregate admin fields remain valid:

- total queue depth;
- total runtime inflight;
- configured and effective inflight capacity.

The bounded admin snapshot exposes active and queued keys with enough state to
explain scheduling decisions:

- pending count;
- active count;
- configured weight;
- current normalized score;
- rejected count by limit.

This is exposed through the bounded admin view. It does not add metric labels
with per-key cardinality.

Inference logs include the normalized fairness key. This makes a retry storm
attributable to its scheduling identity.

## Relationship to asr-pool

`asr-pool` already uses:

- reserved slots for an interactive priority class;
- round-robin between `fairness_key` values inside that class;
- a burst cap between interactive and normal queues.

`llm-pool` has a different policy:

- one fairness class, including anonymous callers;
- pool-owned weights instead of caller-selected priority;
- measured slot-time instead of request count;
- a work-conserving soft per-key inflight cap.

The `asr-pool` burst counter is therefore not copied into V1.

## Implementation boundary

The implementation remains inside the request contract and per-model scheduler
boundary:

- `ResponseRequest` accepts the optional `fairness_key`;
- configuration provides `engine.fairness`;
- `SchedulerJob` retains the normalized key;
- `LoadedModelExecutor` owns per-key FIFO buckets;
- `_run_loop` applies the admission procedure;
- replica completion reports the key and measured slot-time back to the parent
  executor;
- snapshots aggregate all buckets and active jobs;
- shutdown drains every bucket.

Backends do not interpret the fairness key.

## Implemented test coverage

`tests/test_scheduler_fairness.py` covers:

- FIFO order within one key;
- alternating tie-breaks between equal keys;
- weighted long-run slot-time with unequal job durations;
- active elapsed time affecting selection before completion;
- one client borrowing every slot when alone;
- a waiting client receiving the next released slot;
- no idle slots when every key is at the soft cap;
- anonymous and keyed work sharing one scheduler;
- a new key starting at the current minimum score;
- idle-state expiry and cleanup;
- failed active jobs receiving an elapsed-time charge;
- per-key and total queue-limit rejection;
- aggregate queue depth across buckets;
- multi-replica active counts;
- unload draining and failing every pending future.

The tests cover effective capacities `1` and `4`. Capacity `4` exposes the
multi-slot cases that a serial executor cannot.

## Out of scope

- fairness across model executors;
- preemption of active inference;
- reserved interactive slots;
- authentication and gateway-issued identity;
- fairness between users or jobs behind one service;
- per-model weight overrides;
- request rate limiting;
