# Runtime Admin API

This note defines a small admin API for loading, unloading, and inspecting models at runtime.

The goal is to avoid editing `local.json` and restarting the service for routine model management.

It is intentionally a v1 design:

- live runtime control only
- no automatic writes back to `settings.json` or `local.json`
- no arbitrary model definitions via API
- no force unload
- no background job system for model loads

## Purpose

The current service merges `settings.json` and `local.json` into one effective config, then loads enabled models at startup.

The admin API adds a separate live control plane on top of that merged config:

- the merged config tells us which models are known to the service
- the live runtime state tells us which of those models are currently loaded

That distinction must stay explicit in both the API and the UI.

## Core Concepts

### Configured Model Definition

A configured model definition comes from the merged `settings.json + local.json` payload.

This is static process input. It includes fields such as:

- `model_path`
- `backend`
- `device`
- `prompt_format`
- backend-specific settings
- `enabled`

This definition is not modified by the admin API in v1.

### Runtime State

Each configured model also has a live runtime state inside the process.

Allowed states:

- `unloaded`
- `loading`
- `loaded`
- `unloading`
- `failed`

These states are runtime-only and may differ from the original `enabled` value in config.

## State Semantics

### `unloaded`

- the model exists in merged config
- no runtime is currently loaded
- inference requests for this model are rejected
- the model may be loaded through the admin API

### `loading`

- a runtime load has started but is not complete yet
- inference requests for this model are rejected
- duplicate load requests should be treated as idempotent and return current state

### `loaded`

- a runtime exists and may serve inference requests
- the model may be unloaded through the admin API

### `unloading`

- no new inference requests are accepted for this model
- in-flight requests are allowed to finish
- once in-flight requests reach zero, runtime resources are released

### `failed`

- the last load attempt failed
- `last_error` should be retained for inspection
- the model may be loaded again through the admin API

## Request Behavior By Runtime State

For `POST /v1/responses`:

- `loaded`: accept
- `unloaded`: reject
- `loading`: reject
- `unloading`: reject
- `failed`: reject

The error should be explicit and machine-readable.

Suggested error codes:

- `unknown_model`
- `model_not_loaded`
- `model_loading`
- `model_unloading`
- `model_failed`

## Endpoints

### `GET /v1/admin/models`

Returns all known models from merged config together with their live runtime state.

This endpoint is the main UI source of truth.

Suggested response shape:

```json
{
  "models": [
    {
      "name": "google_gemma-4-E2B-it-Q8_0-gguf",
      "resolved_backend": "gguf",
      "configured_enabled": true,
      "runtime_state": "loaded",
      "is_loaded": true,
      "inflight_requests": 0,
      "last_error": null,
      "vram_estimate_mib": 57200,
      "vram_estimate_source": "model_artifact_size",
      "definition": {
        "model_path": "/home/gunnar/models/google_gemma-4-E2B-it-Q8_0/google_gemma-4-E2B-it-Q8_0.gguf",
        "backend": "gguf",
        "device": "cuda",
        "prompt_format": "gemma4_template",
        "enabled": true,
        "gguf_n_gpu_layers": -1,
        "gguf_n_ctx": 4096,
        "gguf_flash_attn": false
      }
    }
  ]
}
```

Notes:

- `configured_enabled` reports what the merged config says
- `runtime_state` reports the live process state
- `vram_estimate_mib` is an approximate per-model VRAM estimate
- `vram_estimate_source` is either `observed_load_delta`, `model_artifact_size`, or `unavailable`
- `definition` should mirror the merged config as closely as practical

### `GET /v1/admin/gpu-memory`

Returns current GPU memory usage (from `nvidia-smi`) and per-model VRAM estimates.

Suggested response shape:

```json
{
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
      "used_mib": 75603,
      "total_mib": 97887,
      "used_over_total": "75603MiB / 97887MiB"
    }
  ],
  "models": [
    {
      "name": "google_gemma-4-E2B-it-Q8_0-gguf",
      "runtime_state": "loaded",
      "is_loaded": true,
      "vram_estimate_mib": 12500,
      "vram_estimate_source": "model_artifact_size"
    },
    {
      "name": "mistral-small-3.2-24b-instruct-2506-gguf",
      "runtime_state": "unloaded",
      "is_loaded": false,
      "vram_estimate_mib": 16800,
      "vram_estimate_source": "model_artifact_size"
    }
  ],
  "error": null
}
```

Notes:

- `used_over_total` matches the compact view you typically read from `nvidia-smi`
- `vram_estimate_mib` for unloaded models is still an estimate, not a reservation
- if `nvidia-smi` is unavailable, `gpus` can be empty and `error` will explain why

### `POST /v1/admin/models/{model_name}/load`

Loads one model that already exists in merged config.

Rules:

- `404` if `model_name` is unknown
- `200` if the model is already `loaded` or `loading`
- transition `unloaded -> loading -> loaded`
- transition `failed -> loading -> loaded`
- if load fails, transition to `failed` and retain `last_error`

Suggested response shape:

```json
{
  "name": "google_gemma-4-E2B-it-Q8_0-gguf",
  "resolved_backend": "gguf",
  "configured_enabled": false,
  "runtime_state": "loaded",
  "is_loaded": true,
  "inflight_requests": 0,
  "last_error": null,
  "vram_estimate_mib": 12340,
  "vram_estimate_source": "observed_load_delta",
  "definition": {
    "model_path": "/home/gunnar/models/google_gemma-4-E2B-it-Q8_0/google_gemma-4-E2B-it-Q8_0.gguf",
    "backend": "gguf",
    "device": "cuda",
    "prompt_format": "gemma4_template",
    "enabled": false
  }
}
```

Notes:

- loading is allowed for configured models even when `configured_enabled` is `false`
- after a successful load, `vram_estimate_source` may switch to `observed_load_delta` if a GPU delta could be measured during load

### `POST /v1/admin/models/{model_name}/unload`

Gracefully unloads one currently loaded model.

Rules:

- `404` if `model_name` is unknown
- `200` if already `unloaded`
- `200` if already `unloading`
- transition `loaded -> unloading -> unloaded`
- new inference requests are rejected once `unloading` starts
- in-flight requests are allowed to finish before resources are released

Suggested response shape:

```json
{
  "name": "google_gemma-4-E2B-it-Q8_0-gguf",
  "resolved_backend": "gguf",
  "configured_enabled": false,
  "runtime_state": "unloaded",
  "is_loaded": false,
  "inflight_requests": 0,
  "last_error": null,
  "vram_estimate_mib": 12340,
  "vram_estimate_source": "observed_load_delta",
  "definition": {
    "model_path": "/home/gunnar/models/google_gemma-4-E2B-it-Q8_0/google_gemma-4-E2B-it-Q8_0.gguf",
    "backend": "gguf",
    "device": "cuda",
    "prompt_format": "gemma4_template",
    "enabled": false
  }
}
```

## Unload And In-Flight Requests

Unload must be graceful in v1.

That means:

- mark the model as `unloading`
- reject new inference requests for that model
- wait for in-flight requests to finish
- release runtime references and backend resources
- mark the model as `unloaded`

The service should track `inflight_requests` per model so unload can wait safely.

## Scheduler Alignment

This admin API should stay compatible with a future external scheduler.

The intended split is:

- scheduler owns external pending queues
- runtime owns backend execution state

That implies the following unload behavior:

- queued but not yet submitted requests: cancel
- already submitted or actively running requests: let them drain in v1

So even after a scheduler exists, `unload` should not mean "kill active GPU work immediately".

It should mean:

- stop new admissions
- cancel scheduler-owned queued work
- drain already submitted runtime work
- then unload the model

## Resource Release Guarantees

For v1, "successful unload" should mean:

- no runtime remains registered for the model
- no new requests can reach that runtime
- no in-flight requests remain
- backend-owned objects are dereferenced
- memory becomes reusable for later loads

The implementation should not promise that every allocator reports zero immediately after unload.

The guarantee is functional reuse, not cosmetic memory counters.

## Documentation Expectations

This API is intended to support a UI, so the implementation should expose:

- stable response models
- OpenAPI descriptions on every admin endpoint
- clear descriptions of runtime states
- clear error codes for rejected inference requests

The UI should be able to render:

- configured vs loaded state
- current lifecycle state
- last load error

## Out Of Scope

This v1 note does not define:

- writes back to config files
- force unload
- active GPU job cancellation
- disk-backed queues
- scheduler fairness policy
- retry policy for failed model loads
