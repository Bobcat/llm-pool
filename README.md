# llm-pool

FastAPI service for local LLM inference with a single `POST /v1/responses` API across CT2, ExLlamaV3, and GGUF/`llama.cpp` backends, plus admin endpoints for loading, unloading, and inspecting models at runtime.

## Overview

- one inference API across CT2, ExLlamaV3, and GGUF/`llama.cpp` backends
- JSON responses and SSE streaming from the same endpoint
- runtime metrics included in inference responses
- admin endpoints for inspecting models and loading or unloading them at runtime
- an in-process scheduler/executor layer in front of inference
- per-model queueing, runtime inflight tracking, and configurable target inflight
- request routing across multiple identical internal replicas for the same model id

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/responses` | Run inference. `stream: false` returns one JSON response; `stream: true` returns Server-Sent Events (SSE). |
| `GET /v1/models` | List currently loaded model ids. |
| `GET /v1/admin/models` | List configured model ids plus aggregate runtime, replica, queue, and load state. |
| `GET /v1/admin/gpu-memory` | Return current GPU memory usage plus per-model VRAM estimates. |
| `POST /v1/admin/models/{model_name}/load` | Load one configured model at runtime with optional runtime-only backend-specific overrides. |
| `POST /v1/admin/models/{model_name}/unload` | Gracefully unload one loaded model. |

See [runtime-admin-api.md](docs/runtime-admin-api.md) for the full admin API.

## Inference Example

Example request:

```json
{
  "model": "google_gemma-4-26B-A4B-it-Q4_K_M-gguf",
  "input": "The weather is pleasant today, and I would like to take a walk in the park after lunch.",
  "instructions": "Translate to Dutch. Return only the translation.",
  "stream": false,
  "decoding": {
    "beam_size": 1,
    "top_k": 1,
    "top_p": 1.0,
    "temperature": 0.1,
    "repetition_penalty": 1.0,
    "max_tokens": 256
  }
}
```

Example response:

```json
{
  "id": "resp_123",
  "object": "response",
  "model": "google_gemma-4-26B-A4B-it-Q4_K_M-gguf",
  "output": [
    {
      "type": "output_text",
      "text": "Het weer is aangenaam vandaag en ik zou na de lunch graag een wandeling in het park willen maken."
    }
  ],
  "output_text": "Het weer is aangenaam vandaag en ik zou na de lunch graag een wandeling in het park willen maken.",
  "metrics": {
    "backend_inference_wall_ms": 138.1,
    "engine_total_wall_ms": 138.6,
    "engine_outside_backend_wall_ms": 0.5,
    "pool_total_wall_ms": 139.0,
    "engine_tokenize_ms": null,
    "gpu_time_to_first_token_ms": null,
    "gpu_generate_total_ms": 138.4,
    "gpu_decode_after_first_token_ms": null,
    "engine_prompt_tokens": 47,
    "engine_output_tokens": 23,
    "engine_tokens_per_second": 166.2
  }
}
```

`POST /v1/responses` supports:

- `stream: false` for one JSON response envelope
- `stream: true` for SSE events: `response.created`, `response.output_text.delta`, `response.metrics`, `response.completed`

Current SSE note:

- `stream: true` still uses the current service-side SSE event path
- it is not yet a true backend-native live token stream from every runtime implementation

## Request Fields

Currently supported API request fields:

| Field | Type | Required | Default if omitted | Notes |
| --- | --- | --- | --- | --- |
| `model` | `string` | yes | none | Must match a currently loaded model id. |
| `input` | `string` | yes | none | Main input text. |
| `instructions` | `string \| null` | no | `null` | Optional high-level guidance. If omitted, the pool falls back to an internal default instruction prompt. |
| `stream` | `boolean` | no | `false` | `false` returns one JSON response; `true` returns SSE events. |
| `decoding` | `object` | no | `{}` | Omitted subfields fall back to `engine.decoding` server defaults. |

Currently supported decoding fields:

| Field | Type | Required | Default if omitted | CT2 | ExLlamaV3 | GGUF | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `beam_size` | `int` | no | server default, usually `1` | used | accepted, ignored | accepted, ignored | ExLlamaV3 and GGUF log it and continue. |
| `top_k` | `int` | no | server default, usually `1` | used | used | used | Sampling control. |
| `top_p` | `float` | no | server default, usually `1.0` | used | used | used | Sampling control. |
| `temperature` | `float` | no | server default, usually `0.1` | used | used | used | Sampling control. |
| `repetition_penalty` | `float` | no | server default, usually `1.0` | used | used | used | Repetition penalty. |
| `max_tokens` | `int` | no | server default, usually `256` | used | used | used | Maximum generated output tokens. |
| `stop` | `list[string]` | no | server default extra stop list, often empty | used | used | used | Optional extra stop strings. Model-internal stop/eos tokens are handled by the pool/backend. |

## Local Overrides

You can keep shared defaults in `config/settings.json` and put machine-local overrides in `config/local.json`.
When present, `local.json` is merged over `settings.json` (override wins per key).

Settings files can also define `service.host`, `service.port`, `service.log_level`, and global `engine.decoding` defaults.

Per model, you can set `model_path`, `device`, `compute_type`, `prompt_format`, `enable_thinking`, `enabled`, `replicas`, `replica_max`, `target_inflight`, and optionally override the backend:

```json
{
  "engine": {
    "backend": "gguf",
    "models": {
      "gemma-4-26B-A4B-it-exl3-5.10bpw": {
        "model_path": "/home/gunnar/models/gemma-4-26B-A4B-it-exl3-5.10bpw",
        "backend": "exllamav3",
        "prompt_format": "gemma4_template",
        "enable_thinking": false,
        "enabled": true,
        "replicas": 1,
        "replica_max": 1,
        "target_inflight": 1,
        "exllama_cache_size": 16384,
        "exllama_cache_quant": "8,8",
        "exllama_tensor_parallel": true,
        "exllama_gpu_split": "24,24"
      },
      "google_gemma-4-26B-A4B-it-Q4_K_M-gguf": {
        "model_path": "/home/gunnar/models/google_gemma-4-26B-A4B-it-Q4_K_M/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "backend": "gguf",
        "device": "cuda",
        "prompt_format": "gemma4_template",
        "enable_thinking": false,
        "enabled": true,
        "replicas": 3,
        "replica_max": 4,
        "target_inflight": 1,
        "gguf_n_gpu_layers": -1,
        "gguf_n_ctx": 4096,
        "gguf_flash_attn": "auto",
        "gguf_type_k": "q8_0",
        "gguf_type_v": "q4_0"
      }
    }
  }
}
```

Notes:
- Models without a `backend` field use the global `engine.backend`.
- `enabled` controls whether a model is loaded by the pool at startup.
- A configured model with `enabled: false` may still be loaded later through the admin API.
- `replicas` is the default replica count that will be started for that model id when it is loaded.
- `replica_max` is the maximum allowed replica count for that model id.
- `target_inflight` is configured per model id and applied per loaded replica through the scheduler.
- `enable_thinking` is an optional per-model template setting for formats that expose a thinking toggle.
- Request-level decoding values override `engine.decoding` defaults when provided.
- ExLlamaV3 models also support `exllama_tp_backend`, `exllama_max_batch_size`, `exllama_max_chunk_size`, `exllama_max_q_size`, and `exllama_max_rq_tokens`.
- GGUF models also support `gguf_n_gpu_layers`, `gguf_n_ctx`, `gguf_flash_attn`, `gguf_type_k`, and `gguf_type_v`.
- The admin API may temporarily override backend-specific load settings at runtime without modifying `settings.json` or `local.json`.
- The exact load override fields, allowed values, defaults, and recommended presets are documented in [runtime-admin-api.md](docs/runtime-admin-api.md).
- ExLlamaV3 dependencies are loaded lazily and required only when an ExLlamaV3 model is configured.
- GGUF dependencies are loaded lazily and required only when a GGUF model is configured.

Optional env vars:
- `LLM_POOL_SETTINGS_PATH`: explicit base settings file path.
- `LLM_POOL_LOCAL_SETTINGS_PATH`: explicit local override file path.

## Replicas

- Clients send only the model id that appears in the API and config.
- A single model id may map to multiple identical internal replicas.
- `/v1/models` returns model ids, not internal replica ids.
- `/v1/admin/models` returns one aggregate row per model id.
- `replicas` is the default replica count for that model id when it is loaded.
- `replica_max` is the maximum allowed replica count for that model id.
- Replicas are only for identical runtime instances. Different context sizes or cache settings require different model ids.

## Timing Metrics

The response `metrics` payload is structured around these boundaries:

- `backend_inference_wall_ms`
  pure backend inference wall time inside the runtime
- `engine_total_wall_ms`
  queue-backed engine/scheduler wall time
- `engine_outside_backend_wall_ms`
  engine wall time outside pure backend inference
- `pool_total_wall_ms`
  service-level wall time inside the `llm-pool` HTTP boundary

This keeps `Inference`, `Engine`, and `Pool` timings separate for callers that want to measure overhead above pure inference.

## Test

```bash
python3 -m unittest discover -s tests
```

## Design Notes

The repo also includes a small set of active design notes:

- [runtime-scheduler-notes.md](docs/runtime-scheduler-notes.md)
  Captures the broader scheduler design space beyond the current in-process implementation.
- [runtime-scheduler-tracker.md](docs/runtime-scheduler-tracker.md)
  Tracks the current scheduler MVP implementation status and remaining deferred work.
- [model-replica-routing-notes.md](docs/model-replica-routing-notes.md)
  Captures the client-visible model and replica-routing semantics.
- [runtime-subprocess-notes.md](docs/runtime-subprocess-notes.md)
  Captures the intended process-isolation model for loaded runtimes and how that should fit behind the same scheduler/runtime adapter boundary.

## Acknowledgments

This pool builds on a number of excellent upstream projects:

- FastAPI
- Uvicorn
- Pydantic
- CTranslate2
- Transformers
- ExLlamaV3
- llama-cpp-python
- llama.cpp
