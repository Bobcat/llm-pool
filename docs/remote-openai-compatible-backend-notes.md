# Remote OpenAI-Compatible Backend Notes

This note captures a possible remote backend for `llm-pool`.

It is a design note, not an implementation spec.

Status: proposed.

## Purpose

`llm-pool` currently routes requests to backends that act on locally loaded model runtimes.

That works well for models that fit on the local GPU, but some useful models may only be practical through an external provider.

The goal is to add a backend that:

- uses an API key instead of local model weights
- routes to an upstream OpenAI-compatible model API
- keeps the existing public `llm-pool` model contract
- keeps the existing scheduler, queue, replica, admin, and metrics shape as much as possible

The important idea is:

- local backends load model weights
- remote backends activate a configured upstream route

Both should still look like loaded public models to the rest of `llm-pool`.

## Working Backend Name

Suggested working name:

- `openai_compatible`

This name is intentionally provider-neutral.

It should not imply that the upstream provider must be OpenAI itself. It should only mean that the upstream endpoint accepts an OpenAI-compatible request/response shape.

## Core Model

One configured `llm-pool` model remains one public model id.

Example:

```json
{
  "engine": {
    "models": {
      "frontier-large": {
        "backend": "openai_compatible",
        "remote_api_kind": "chat_completions",
        "remote_base_url": "https://api.example.com/v1",
        "remote_api_key_env": "EXAMPLE_API_KEY",
        "remote_model": "provider-large-model",
        "enabled": false
      }
    }
  }
}
```

Clients still send:

```json
{
  "model": "frontier-large",
  "input": "Explain this document.",
  "stream": false
}
```

The caller does not send arbitrary upstream model names.

For V1, the remote backend should target OpenAI-compatible Chat Completions:

```json
{
  "remote_api_kind": "chat_completions"
}
```

Responses-style APIs can be added later as a separate adapter path if needed.

The configured public model id controls:

- scheduler queue ownership
- admin visibility
- load/unload state
- cost policy
- upstream model routing
- provider credentials

This preserves the current `llm-pool` rule:

- different runtime profiles are different public models
- hidden routing between heterogeneous profiles is not introduced

## Non-Goals

This design does not introduce:

- arbitrary request-time routing to any upstream model
- automatic provider/model discovery
- provider-specific feature parity
- billing reconciliation against provider invoices
- retry policies that may silently increase cost
- persistent distributed accounting across multiple `llm-pool` processes
- true token-by-token streaming in the first version

## Config Shape

Remote backend fields should be explicit.

They should not be squeezed into `model_path`.

Possible V1 fields:

- `remote_api_kind`
  upstream API shape; V1 should support `"chat_completions"`
- `remote_base_url`
  upstream API base URL
- `remote_api_key_env`
  environment variable containing the API key
- `remote_model`
  upstream model id sent to the provider
- `remote_timeout_s`
  request timeout in seconds
- `remote_health_check`
  default `"config_only"`; validates config and API key env without making an upstream completion call
- `remote_max_retries`
  default `0` for V1
- `remote_price_input_per_1m`
  configured input-token price per one million tokens
- `remote_price_output_per_1m`
  configured output-token price per one million tokens
- `remote_price_currency`
  default `"USD"`
- `remote_budget_daily`
  optional process-local daily budget for this public model
- `remote_budget_request_max`
  optional maximum estimated cost for one request
- `remote_cost_ledger_path`
  optional JSONL ledger path
- `remote_reservation_stale_after_s`
  optional window after which open reservations are considered stale for budget checks

Price fields use per-million-token units because that matches the common provider-facing pricing format and reduces decimal mistakes in config.

`model_path` should probably remain required only for local backends.

If changing that assumption is too broad for the first patch, a temporary compatibility decision should be explicit in the implementation tracker rather than hidden in the backend.

## Load And Unload Semantics

For a remote backend, `load` does not load weights into VRAM.

V1 `load` should mean:

- validate that required config fields are present
- validate that the API key environment variable exists
- optionally perform a cheap upstream health/model check
- register the scheduler executor for the public model
- transition runtime state to `loaded`

The default health check should be config-only:

- required remote fields are present
- the API key environment variable is present

Some providers do not expose a useful `/models` endpoint, and some API keys only have inference access.
`load` should not make a real completion call unless an explicit future setting opts into that behavior.

V1 `unload` should mean:

- reject new requests for that public model
- cancel scheduler-owned queued work
- let already submitted upstream calls drain
- unregister the scheduler executor
- transition runtime state to `unloaded`

This matches the existing scheduler rule:

- queued work is cancelled
- runtime-submitted work is drained

## Replicas

Remote public models should keep the existing replica semantics.

For a local backend, a replica usually means another loaded runtime instance.

For a remote backend, a replica means another identical route handle with the same:

- upstream base URL
- upstream model id
- credential source
- timeout and cost policy

The scheduler can still dispatch across replicas using the existing least-inflight plus round-robin policy.

Remote replicas do not duplicate local VRAM use, but they may increase:

- upstream concurrency
- provider rate-limit pressure
- spend rate

Therefore, remote replica count and `target_inflight` should be treated as cost and rate-limit knobs, not just throughput knobs.

## Scheduler Boundary

The scheduler should not learn provider details.

It should continue to see:

- a loaded public model executor
- one or more replica registrations
- a `complete_fn`
- a runtime capability

The remote backend should own:

- HTTP request construction
- provider authentication
- upstream response parsing
- upstream usage extraction
- provider error mapping into stable backend errors

The pool/router/control plane should own:

- model admission
- budget admission
- scheduler enqueue
- admin state
- aggregate metrics

## Request Flow

Suggested V1 flow:

1. FastAPI validates the normal `ResponseRequest`.
2. `ModelRouterEngine` verifies that the public model is configured and loaded.
3. If request or caller policy says remote execution is not allowed, reject before enqueue.
4. If the resolved backend is remote and cost control is enabled:
   - estimate maximum request cost
   - reserve budget in the local ledger
   - reject before enqueue if the request exceeds policy
5. The existing scheduler enqueues the request.
6. The selected remote replica calls the upstream API.
7. The remote backend extracts text, usage, and provider timing if available.
8. The pool settles or releases the cost reservation.
9. The response returns through the existing response envelope.

This preserves synchronous HTTP behavior:

- enqueue
- wait
- return completed response

## Decoding Mapping

Some current decoding fields map cleanly to common OpenAI-compatible APIs:

- `temperature`
- `top_p`
- `max_tokens`
- `stop`

Some fields are not consistently supported:

- `beam_size`
- `top_k`
- `repetition_penalty`

V1 should accept the existing request schema but handle unsupported fields explicitly.

Possible V1 rule:

- map supported fields
- ignore unsupported fields only with a structured log line
- add a metrics counter or response warning later if callers need visible feedback
- do not add provider-specific request fields yet

If provider-specific options are needed later, they should be introduced as explicit config or request extension fields rather than quietly overloading existing decoding fields.

## Streaming

The current service-side SSE path is not true backend-native streaming for every runtime.

V1 remote backend should not try to solve streaming unless that is the explicit task.

Acceptable V1 behavior:

- `stream: false` returns one JSON response
- `stream: true` keeps the current service-side SSE envelope after completion

Future work can add true upstream token streaming behind the scheduler/runtime boundary.

That should be a separate design because it affects:

- scheduler completion semantics
- cost settlement timing
- cancellation behavior
- subprocess transport if runtime isolation is later introduced

## Cost Control

Remote calls can spend money, so provider-side limits should not be the only guardrail.

`llm-pool` should be able to reject a request before it leaves the machine.

V1 cost control should be small and local:

- static price config per public model
- preflight maximum-cost estimate
- append-only JSONL ledger
- process-local budget checks
- post-response actual usage and cost settlement

The provider remains the final billing authority, but the pool owns local admission policy.

## Cost Estimate

Preflight cost is an estimate.

For V1, the conservative estimate can use:

- estimated prompt tokens
- configured or requested `max_tokens`
- configured input/output token prices per one million tokens

Actual cost should use upstream `usage` when available:

- prompt/input tokens
- completion/output tokens
- total tokens if separate counts are unavailable

If upstream usage is missing, the backend should report that clearly and either:

- estimate actual usage locally
- or mark actual cost as unavailable

For budgeted models, usage missing after an upstream submission should fail closed.

Acceptable V1 behavior:

- settle conservatively against the reserved maximum
- or keep the reservation counted until it expires as stale

It should not be released as free work if the provider may have processed the request.

The note does not require provider-specific tokenizers in V1.

## JSONL Cost Ledger

A ledger does not need SQLite.

V1 can use an append-only JSONL file.

Example events:

```json
{"ts":"2026-05-06T19:20:00Z","event":"reserved","request_id":"resp_123","model":"frontier-large","estimated_max_cost":0.12,"currency":"USD"}
{"ts":"2026-05-06T19:20:08Z","event":"settled","request_id":"resp_123","model":"frontier-large","input_tokens":1200,"output_tokens":400,"actual_cost":0.031,"currency":"USD"}
{"ts":"2026-05-06T19:30:00Z","event":"abandoned","request_id":"resp_456","model":"frontier-large","reason":"stale_reservation_after_restart"}
```

Useful event kinds:

- `reserved`
  budget was reserved before enqueue or upstream submission
- `settled`
  upstream call completed and actual usage/cost is known
- `released`
  reserved budget was released because the request did not reach the upstream provider
- `failed_before_submit`
  request failed before reaching the upstream provider
- `failed_after_submit`
  request failed after upstream submission; provider cost may still exist
- `failed_usage_unknown`
  request may have reached the upstream provider, but usage was unavailable
- `abandoned`
  stale open reservation was marked as no longer active for budget admission

The ledger should be append-only.

Budget checks should count:

- settled cost inside the active budget window
- open reservations inside the active budget window

That prevents many concurrent requests from passing the same budget check and overspending together.

Open reservations need crash recovery semantics.

If the process writes `reserved` and then crashes before writing a terminal event, the reservation is safe but may block future budget unnecessarily.

V1 should handle that explicitly with one of these rules:

- count open reservations only until `remote_reservation_stale_after_s`
- or provide an admin/maintenance path that appends `abandoned` events for stale reservations

Stale reservation handling is a local admission rule.
It is not proof that the provider did or did not bill the request.

## Ledger Locking

The JSONL ledger needs a lock around budget admission.

The lock should cover:

1. reading current usage and open reservations
2. deciding whether the new request is allowed
3. appending the new `reserved` event

The lock does not need to cover the upstream API call.

After completion or failure, the service appends `settled`, `released`, `failed_before_submit`, `failed_after_submit`, or `failed_usage_unknown`.

This is enough for a single local `llm-pool` process.

It is not a distributed accounting system.

## Budget Scope

V1 budget scope can be per public model.

That is enough to answer:

- how much may this configured remote model spend today?
- what is the maximum estimated cost of one request to this model?

Future scopes may include:

- per provider
- per client
- per project
- per API key
- per time window beyond daily

Those should not be added until there is a concrete caller identity model.

## Remote Admission Policy

Remote execution should be opt-in for callers that handle private local workloads.

This is separate from cost admission.

Examples of workloads that may need this:

- transcripts
- customer text
- audio-derived text
- private documents

The exact request or app-level field is not decided here.

Possible shapes:

```json
{
  "allow_remote": false
}
```

or:

```json
{
  "local_only": true
}
```

The important V1 rule is:

- a caller must be able to prevent a request from being routed to a remote backend before scheduler enqueue

Apps that process private workloads should default to local-only unless remote execution is explicitly allowed.

This prevents accidental cloud routing just because a selected public model id resolves to a remote backend.

## Admin And Metrics

Remote models should appear in admin output like other configured models.

Useful remote-specific admin fields later:

- `remote_api_kind`
- `remote_base_url`
- `remote_model`
- `remote_api_key_env`
- `remote_timeout_s`
- `remote_health_check`
- `remote_budget_daily`
- `remote_budget_remaining_estimate`
- `remote_cost_ledger_path`
- `remote_reservation_stale_after_s`

Secrets should not be returned.

The existing inference metrics can be extended later with:

- upstream request wall time
- upstream prompt tokens
- upstream output tokens
- estimated max cost
- actual cost
- cost currency
- failed-after-submit or usage-unknown status
- cost ledger event id or sequence number

V1 can start by storing cost data in the ledger and only adding response/admin fields when needed by a UI or caller.

## Error Semantics

Remote backend errors should be machine-readable where possible.

Useful error classes:

- missing API key environment variable
- invalid remote config
- budget exceeded before enqueue
- request max cost exceeded
- remote execution disallowed by request or caller policy
- upstream timeout
- upstream authentication failure
- upstream rate limit
- upstream invalid request
- upstream server error
- upstream response parse failure

Retries should default to off in V1.

Reason:

- retries can duplicate spend
- rate-limit behavior differs by provider
- retry policy belongs with explicit cost and error semantics

## Security

API keys should be read from environment variables.

They should not be stored directly in config files.

Admin responses and logs should include:

- the environment variable name
- never the API key value

Request and response logs should be careful with prompt and output text.

Cost logging can avoid prompt text entirely by recording:

- request id
- public model
- token counts
- cost estimates
- settlement status

## Implementation Order

Suggested phased implementation:

1. Add remote config fields and parse them into `ModelSettings`.
2. Add an `OpenAICompatibleEngine` that can load configured remote routes.
3. Implement the `chat_completions` request/response adapter.
4. Register remote replicas through the existing scheduler path.
5. Implement non-streaming upstream completion with timeout and retries defaulting to `0`.
6. Add request-level remote admission policy before enqueue.
7. Map upstream usage into existing metrics where possible.
8. Add JSONL cost ledger reservations, settlements, failed-after-submit handling, and stale reservation handling.
9. Surface cost/admin fields only when there is a concrete UI/API consumer.
10. Consider true upstream streaming as a separate phase.

## Open Questions

- Should budget admission happen in `ModelRouterEngine`, or in a small cost-policy helper owned by it?
- Should `ResponseRequest` or scheduler jobs carry the final response id so ledger events can use the same id as the HTTP response?
- Should the remote-admission flag be request-level, app-level, or both?
- Should `failed_usage_unknown` settle conservatively immediately or keep the reservation open until it expires?
- Should `remote_max_retries` exist in config while defaulting to `0`, or be omitted until retries are implemented?

## Practical Conclusion

The remote backend fits the current architecture if it is treated as another loaded runtime behind a public model id.

The key constraints are:

- keep public model ids configured, not arbitrary per request
- keep scheduler policy outside provider-specific code
- keep credentials and upstream model ids in config
- make local cost admission explicit before requests leave the machine
- start with an append-only JSONL ledger before reaching for a database

That gives `llm-pool` a way to use larger frontier models without abandoning its current scheduler, admin, replica, and response contracts.
