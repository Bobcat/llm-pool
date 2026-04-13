# llm-pool

FastAPI service for local LLM inference behind one generic `POST /v1/responses` contract.

Current scope:
- generic request schema with per-request instructions and decoding params
- JSON response mode
- SSE streaming mode
- `GET /v1/models` endpoint for loaded model discovery
- admin endpoints for runtime inspection and live model load/unload
- per-request runtime metrics in the response
- CT2, ExLlamaV3, and GGUF/`llama.cpp` engines behind the same API contract
- per-model backend routing inside one llm-pool instance
- includes model serving and routing; queue/scheduler and runtime-isolation work is tracked in the design notes below

## Endpoints

`POST /v1/responses`

- `stream: false` returns a JSON response envelope.
- `stream: true` returns SSE events: `response.created`, `response.output_text.delta`, `response.metrics`, `response.completed`.

`GET /v1/models`

- returns the currently loaded models.

`GET /v1/admin/models`

- returns all configured models plus their live runtime state.
- includes fields such as `configured_enabled`, `runtime_state`, `inflight_requests`, `last_error`, and the resolved model definition.

`GET /v1/admin/gpu-memory`

- returns current GPU memory usage from `nvidia-smi` plus per-model VRAM estimates.

`POST /v1/admin/models/{model_name}/load`

- live-loads one configured model without modifying `settings.json` or `local.json`.
- returns `404` for an unknown model, `409` if the model is currently unloading, and `500` if the load attempt fails.

`POST /v1/admin/models/{model_name}/unload`

- gracefully unloads one configured model at runtime.
- new inference requests are rejected once unloading starts, and in-flight requests are allowed to finish before cleanup.
- returns `404` for an unknown model and `409` if the model is currently loading.

The original design note for this control plane is in [runtime-admin-api.md](docs/runtime-admin-api.md).

Example request:

```json
{
  "model": "eurollm-9b-ct2-int8",
  "input": "Hello world",
  "instructions": "Translate to Dutch.",
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
  "model": "eurollm-9b-ct2-int8",
  "output": [
    {
      "type": "output_text",
      "text": "Hallo wereld"
    }
  ],
  "output_text": "Hallo wereld",
  "metrics": {
    "engine_tokenize_ms": 2.7,
    "gpu_time_to_first_token_ms": 38.1,
    "gpu_generate_total_ms": 144.5,
    "gpu_decode_after_first_token_ms": 106.4,
    "engine_prompt_tokens": 22,
    "engine_output_tokens": 5,
    "engine_tokens_per_second": 34.6
  }
}
```

## Request Fields

Currently supported API request fields:

| Field | Type | Required | Default if omitted | Notes |
| --- | --- | --- | --- | --- |
| `model` | `string` | yes | none | Must match a currently loaded configured model. |
| `input` | `string` | yes | none | Main user input text. |
| `instructions` | `string \| null` | no | `null` | If omitted, the pool falls back to an internal default instruction prompt. |
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

Per model, you can set `model_path`, `device`, `compute_type`, `prompt_format`, `enable_thinking`, `enabled`, and optionally override the backend:

```json
{
  "engine": {
    "backend": "ct2",
    "models": {
      "eurollm-9b-ct2-int8": {
        "model_path": "/models/eurollm-ct2",
        "backend": "ct2",
        "enabled": true
      },
      "gemma-4-26B-A4B-it-exl3-5.10bpw": {
        "model_path": "/home/gunnar/models/gemma-4-26B-A4B-it-exl3-5.10bpw",
        "backend": "exllamav3",
        "prompt_format": "gemma4_template",
        "enable_thinking": false,
        "enabled": true,
        "exllama_cache_size": 16384,
        "exllama_cache_quant": "8,8",
        "exllama_tensor_parallel": true,
        "exllama_gpu_split": "24,24"
      },
      "google_gemma-4-E2B-it-Q8_0-gguf": {
        "model_path": "/home/gunnar/models/google_gemma-4-E2B-it-Q8_0/google_gemma-4-E2B-it-Q8_0.gguf",
        "backend": "gguf",
        "device": "cuda",
        "prompt_format": "gemma4_template",
        "enable_thinking": false,
        "enabled": true,
        "gguf_n_gpu_layers": -1,
        "gguf_n_ctx": 4096,
        "gguf_flash_attn": false
      }
    }
  }
}
```

Notes:
- Models without a `backend` field use the global `engine.backend`.
- `enabled` controls whether a model is loaded by the pool at startup.
- A configured model with `enabled: false` may still be loaded later through the admin API.
- `enable_thinking` is an optional per-model template setting for formats that expose a thinking toggle.
- Request-level decoding values override `engine.decoding` defaults when provided.
- ExLlamaV3 models also support `exllama_tp_backend`, `exllama_max_batch_size`, `exllama_max_chunk_size`, `exllama_max_q_size`, and `exllama_max_rq_tokens`.
- GGUF models also support `gguf_n_gpu_layers`, `gguf_n_ctx`, and `gguf_flash_attn`.
- ExLlamaV3 dependencies are loaded lazily and required only when an ExLlamaV3 model is configured.
- GGUF dependencies are loaded lazily and required only when a GGUF model is configured.

Optional env vars:
- `LLM_POOL_SETTINGS_PATH`: explicit base settings file path.
- `LLM_POOL_LOCAL_SETTINGS_PATH`: explicit local override file path.

## Test

```bash
python3 -m unittest discover -s tests
```

## Design Notes

The repo also includes a small set of active design notes for work that is intended but not fully implemented yet:

- [runtime-scheduler-notes.md](docs/runtime-scheduler-notes.md)
  Captures the intended queue and scheduler boundary, including the small runtime adapter interface that backends should implement.
- [runtime-subprocess-notes.md](docs/runtime-subprocess-notes.md)
  Captures the intended process-isolation model for loaded runtimes and how that should fit behind the same runtime adapter boundary.

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
