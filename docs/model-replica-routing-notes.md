## Model Replica Routing Notes

### Status

Design note. This document describes the intended public-model and replica-routing shape for `llm-pool`.

Current reality note:

- the first aggregate public-model replica MVP is now implemented
- one public model may load multiple identical replicas
- admin remains aggregate per public model
- local runtime capability is still clamped to one in-flight request per replica except where a backend explicitly reports more capacity
- `llama_server` participates as another backend behind the same aggregate public-model semantics; native llama-server multi-inflight behavior is not exposed as a scheduler capacity yet
- live resizing, per-replica admin rows, and per-replica unload remain out of scope

This is separate from `runtime-scheduler-notes.md` because replicas affect more than scheduler internals:

- config shape
- public model naming
- admin semantics
- load/unload behavior
- UI behavior in `llm-workbench`

### Problem

We want to scale some small models by running multiple identical runtime instances while keeping the client contract stable.

Clients should continue to send a public model id such as:

- `gemma_translate`
- `gemma_topics_longctx`

They should not need to know about internal replica ids such as:

- `gemma_translate#1`
- `gemma_translate#2`

At the same time, some use cases require a different runtime profile for the same underlying model family. For example:

- translation requests may use a shorter context
- long topics extraction may use a larger context and possibly different cache settings

Those are not replicas of the same public model. They are different public models.

### Terms

#### Public model

The model id sent by clients in `POST /v1/responses`.

Examples:

- `gemma_translate`
- `gemma_topics_longctx`

#### Replica

An identical runtime instance behind a single public model.

Replica means:

- same backend
- same weights / artifact
- same load-time settings
- same context size
- same KV/cache settings

If those differ, the runtime is not a replica. It belongs to a different public model.

### Non-goal

This note does not introduce hidden routing between heterogeneous runtime profiles behind one public model.

In particular:

- a short-context translation profile
- and a long-context topics profile

should not both sit behind the same public model id with implicit pool-side request classification.

The caller should choose between different public models when the runtime profile differs.

### Config shape

For MVP, one configured model entry remains one public model.

Additional per-model fields:

- `replicas`
  desired replica count to start when the model is loaded
  default: `1`
- `replica_max`
  maximum allowed replica count for the model
  default: `1`

If `replica_max == 1`, the model is effectively non-replicated.

Example:

```json
{
  "engine": {
    "models": {
      "gemma_translate": {
        "model_path": "/models/gemma-2b-q8.gguf",
        "backend": "gguf",
        "gguf_n_ctx": 4096,
        "replicas": 3,
        "replica_max": 4
      },
      "gemma_topics_longctx": {
        "model_path": "/models/gemma-2b-q8.gguf",
        "backend": "gguf",
        "gguf_n_ctx": 16384,
        "replicas": 1,
        "replica_max": 1
      }
    }
  }
}
```

### Internal runtime ids

Internally, each loaded runtime still needs a concrete unique id.

Suggested form:

- `gemma_translate#1`
- `gemma_translate#2`
- `gemma_translate#3`

These are internal runtime ids, not public client model ids.

### Client API

#### `/v1/models`

Should continue to return public model ids only.

Examples:

- `gemma_translate`
- `gemma_topics_longctx`

It should not return replica ids.

#### `POST /v1/responses`

The caller provides a public model id.

The pool resolves that public model to one of its loaded replicas.

### Admin semantics

For MVP, admin remains public-model-oriented.

There is one admin row per public model, not per replica.

The row should expose aggregate runtime state for the whole model.

Suggested additional admin fields:

- `replicas`
  configured desired count
- `replica_max`
  configured maximum
- `loaded_replicas`
  current loaded replica count
- `runtime_inflight`
  aggregate inflight across loaded replicas
- `queue_depth`
  public-model queue depth

### UI semantics

In `llm-workbench` `#llm-pool-models`:

- one visible row per public model
- no per-replica rows in MVP

When `replica_max == 1`:

- do not show replica info in the row
- do not show a replica dropdown in the details section

When `replica_max > 1`:

- show a `Replica count` dropdown in the details section
- the dropdown is disabled while the model is loaded or loading

### Load and unload behavior

MVP keeps aggregate model semantics simple.

#### Load

- load always starts exactly the selected `replicas` count
- the count is only chosen while the model is unloaded or failed

#### Unload

- unload stops all replicas of the public model

#### No live resizing in MVP

The following are intentionally out of scope:

- scale from 1 to 3 while already loaded
- unload only replica `#2`
- keep some replicas while replacing others

### Load failure semantics

For MVP, loading multiple replicas should be all-or-nothing.

If the selected replica count is `3`, then:

- either `3/3` replicas load successfully
- or the load fails and any partially started replicas are torn down

This avoids ambiguous aggregate states such as partially loaded models in the first implementation.

### Scheduler ownership

The scheduler queue should remain attached to the public model, not to individual replicas.

Flow:

1. request arrives for public model `X`
2. request is queued on `X`
3. scheduler picks one eligible loaded replica of `X`
4. request is submitted to that replica

### Replica selection policy

For MVP and for future `effective_target_inflight > 1`, the scheduler should already be replica-aware.

Proposed rule:

- choose the eligible replica with the lowest current inflight count
- if multiple replicas tie, use round-robin tie-breaking

This explicitly does not attempt:

- request duration estimation
- prompt-size estimation
- backend-specific latency prediction

If later replicas can have different capacities, this policy may be upgraded to use normalized occupancy:

- `runtime_inflight / effective_target_inflight`

But MVP can start with least-inflight plus round-robin ties.

### Relationship to scheduler note

This note assumes the scheduler note direction still holds:

- scheduler policy remains outside backend internals
- runtime adapters remain per loaded runtime
- queue ownership remains in the scheduler layer

The scheduler note should therefore be read together with this replica-routing note.

### Backend-native concurrency vs replicas

Replicas are the current way to serve concurrent requests for a single public
model: load the weights N times, one runtime per replica, each runtime
serializing its own work. This is a quick, predictable workaround, not the
end-state architecture. Its cost is linear in VRAM: K concurrent requests need
K copies of the weights.

The more efficient path is backend-native concurrency: one weight copy serving
many in-flight requests through continuous batching. vLLM already works this
way. llama.cpp can also do it, but the GGUF backend does not expose it yet — it
constructs the runtime with `n_ctx` only and serializes generation with a lock,
so it is effectively single-sequence today.

#### The two axes

Concurrency always involves the same three quantities: per-request length, a
total KV cache pool, and how many requests run at once. vLLM and llama.cpp
expose them inversely:

| | per-request length | total KV pool | concurrency |
| --- | --- | --- | --- |
| vLLM | `max_model_len` (set directly) | `kv_cache_memory_bytes` (set directly) | derived = pool / per-request |
| llama.cpp | derived = `n_ctx / n_parallel` | `n_ctx` (in tokens) | `n_parallel` (set directly) |

So llama.cpp is not fundamentally single-axis. `n_ctx` is the total cache in
tokens shared across `n_parallel` slots, and each slot gets `n_ctx / n_parallel`
tokens of context.

Example — 5 concurrent users, 32k context each, one weight copy:

```text
n_ctx      = 5 * 32768 = 163840
n_parallel = 5
-> each slot gets 163840 / 5 = 32768 tokens
```

#### Intended direction

When GGUF concurrency is taken seriously, the backend should gain an
`n_parallel` (slot count) knob, set `n_ctx = n_parallel * per_request_ctx`, and
drop the per-runtime serialization lock so slots decode concurrently. That
backend-native path is the proper replacement for the replica workaround for
GGUF; replicas remain useful where a single weight copy cannot be shared (for
example heterogeneous load profiles, or backends without batching support).

This is a backend-internals concern and stays behind the same runtime adapter
boundary; it does not change the public-model or replica contract described
above.

### MVP boundary

Included in MVP:

- one public model entry may load multiple identical replicas
- public model queue with replica-aware dispatch
- aggregate admin row per public model
- replica count configurable only while unloaded
- aggregate load and aggregate unload

Deferred:

- per-replica admin rows
- per-replica unload controls
- heterogeneous profiles behind one public model
- live scaling while loaded
- partial-load states
- predictive routing
