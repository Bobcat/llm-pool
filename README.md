# llm-pool

FastAPI service that exposes one `POST /v1/responses` API across local CT2, ExLlamaV3, GGUF/`llama.cpp`, in-process vLLM, and OpenAI-compatible remote Chat Completions backends. It supports text and image (multimodal) input, and includes runtime metrics, per-model scheduling, replica routing, and admin endpoints for loading, unloading, and inspecting models.

## Index

- [Overview](#overview)
- [HTTP API](#http-api)
- [Inference Example](#inference-example)
- [Request Fields](#request-fields)
- [Multimodal Input](#multimodal-input)
- [Local Overrides](#local-overrides)
- [vLLM Backend](#vllm-backend)
- [Remote Backends](#remote-backends)
- [Replicas](#replicas)
- [Timing Metrics](#timing-metrics)
- [Test](#test)
- [Design Notes](#design-notes)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

- one inference API across local and remote model backends
- local CT2, ExLlamaV3, GGUF/`llama.cpp`, and in-process vLLM runtime adapters
- OpenAI-compatible remote Chat Completions adapter
- text and image (multimodal) input on backends that support it (vLLM, remote)
- JSON responses and SSE streaming from the same endpoint
- runtime metrics included in inference responses
- admin endpoints for inspecting models and loading or unloading them at runtime
- an in-process scheduler/executor layer in front of inference
- per-model queueing, runtime inflight tracking, and configurable target inflight
- request routing across multiple identical internal replicas for the same model id
- request-level `allow_remote` gate for external or paid model calls

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/responses` | Run inference. `stream: false` returns one JSON response; `stream: true` returns Server-Sent Events (SSE). |
| `GET /v1/models` | List currently loaded model ids. |
| `GET /v1/admin/models` | List configured model ids plus aggregate runtime, replica, queue, load state, and `capabilities.modalities` (which input modalities each model accepts). |
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
| `input` | `string \| array` | yes | none | Main input. A plain string for text, or an array of content items for multimodal input. See [Multimodal Input](#multimodal-input). |
| `instructions` | `string \| null` | no | `null` | Optional high-level guidance. If omitted, the pool falls back to an internal default instruction prompt. Ignored by `translategemma_template`; omit it there. |
| `source_lang_code` | `string \| null` | no | `null` | Required for models using `prompt_format: "translategemma_template"`. |
| `target_lang_code` | `string \| null` | no | `null` | Required for models using `prompt_format: "translategemma_template"`. |
| `stream` | `boolean` | no | `false` | `false` returns one JSON response; `true` returns SSE events. |
| `allow_remote` | `boolean` | no | `false` | Required for requests to `openai_compatible` remote models. |
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

Remote OpenAI-compatible models map `temperature`, `top_p`, `max_tokens`, and `stop` to Chat Completions requests. They accept `beam_size`, `top_k`, and `repetition_penalty` for request schema compatibility, but log and ignore them.

## Multimodal Input

`input` is polymorphic. A plain string is text input and behaves exactly as before. An array of content items enables multimodal input using the OpenAI-compatible content shape:

```json
{
  "model": "qwen2.5-vl-3b",
  "instructions": "You describe images.",
  "input": [
    { "type": "text", "text": "What does this sign say? Answer in English." },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBORw0KGgo..." } }
  ],
  "stream": false,
  "decoding": { "temperature": 0.2, "max_tokens": 256 }
}
```

Content item types:

- `{ "type": "text", "text": "..." }`
- `{ "type": "image_url", "image_url": { "url": "...", "detail": "auto" } }`
  The `url` may be a `data:image/...;base64,...` URL, a `file://` path, or an `http(s)://` URL.

Notes:

- Which models accept images is reported per model as `capabilities.modalities` on `GET /v1/admin/models`. A model declares image support with `"modalities": ["text", "image"]` in its config; the default is `["text"]`.
- Text-only backends (CT2, ExLlamaV3, GGUF) reject image content with a `modality_unsupported` 400 error. Image input is currently served by the vLLM backend (local vision models) and by remote OpenAI-compatible vision models.
- A text-only array (only `text` items) is accepted by any backend and is concatenated to a single prompt, so string and text-array input are equivalent.

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
- OpenAI-compatible remote models support `remote_api_kind`, `remote_base_url`, `remote_api_key_env`, `remote_model`, `remote_timeout_s`, `remote_health_check`, `remote_max_retries`, and `remote_thinking`.
- vLLM models support `vllm_model`, `vllm_dtype`, `vllm_gpu_memory_utilization`, `vllm_max_model_len`, `vllm_tensor_parallel_size`, `vllm_trust_remote_code`, `vllm_enforce_eager`, `vllm_limit_mm_per_prompt`, `vllm_mm_processor_kwargs`, and the speculative-decoding fields `vllm_speculative_method`, `vllm_speculative_model`, and `vllm_num_speculative_tokens`. See [vLLM Backend](#vllm-backend).
- `modalities` declares which input modalities a model accepts, e.g. `["text", "image"]`. It defaults to `["text"]` and is surfaced as `capabilities.modalities` on `GET /v1/admin/models`.
- Requests to GGUF models using `prompt_format: "translategemma_template"` must include `source_lang_code` and `target_lang_code`; put only the source text in `input` and omit `instructions`.
- The admin API may temporarily override backend-specific load settings at runtime without modifying `settings.json` or `local.json`.
- The exact load override fields, allowed values, defaults, and recommended presets are documented in [runtime-admin-api.md](docs/runtime-admin-api.md).
- ExLlamaV3 dependencies are loaded lazily and required only when an ExLlamaV3 model is configured.
- GGUF dependencies are loaded lazily and required only when a GGUF model is configured.

TranslateGemma request example:

```json
{
  "model": "translategemma-12b-it-q5-k-m-gguf",
  "input": "Ach, hij is gewoon een ouwe brombeer, maar hij bedoelt het goed.",
  "source_lang_code": "nl",
  "target_lang_code": "en"
}
```

Optional env vars:
- `LLM_POOL_SETTINGS_PATH`: explicit base settings file path.
- `LLM_POOL_LOCAL_SETTINGS_PATH`: explicit local override file path.

## vLLM Backend

The `vllm` backend runs the vLLM Python engine in-process via `AsyncLLMEngine`, on a dedicated event-loop thread. It does not start a separate `vllm serve` process. It supports text and image input, making it the local path for vision-language models.

Example (local vision-language model):

```json
{
  "engine": {
    "models": {
      "qwen2.5-vl-3b": {
        "backend": "vllm",
        "model_path": null,
        "prompt_format": "generic",
        "modalities": ["text", "image"],
        "vllm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory_utilization": 0.7,
        "vllm_max_model_len": 16384,
        "vllm_limit_mm_per_prompt": {"image": 4},
        "vllm_mm_processor_kwargs": {"max_pixels": 4014080},
        "enabled": true
      }
    }
  }
}
```

vLLM backend notes:

- `model_path` is not required; the model is identified by `vllm_model` (a Hugging Face model id or local path).
- `vllm_max_model_len` must be large enough for the text tokens plus the image tokens. A high-resolution image can expand to many thousands of vision tokens; cap it with `vllm_mm_processor_kwargs` (for Qwen2.5-VL, `max_pixels`) to bound vision tokens and avoid exceeding the context length. Lower `max_pixels` reduces tokens at the cost of fine-text legibility.
- `vllm_limit_mm_per_prompt` caps the number of multimodal items of each kind per request, e.g. `{"image": 4}`.
- The speculative-decoding fields (`vllm_speculative_method`, `vllm_speculative_model`, `vllm_num_speculative_tokens`) are wired through to vLLM's `speculative_config`. The Gemma 4 MTP path is not yet verified; see [gemma4-mtp-vllm-notes.md](docs/gemma4-mtp-vllm-notes.md).
- vLLM dependencies are heavy (PyTorch, flashinfer, CUDA libraries) and are imported lazily, only when a vLLM model is loaded.

Blackwell (SM 12.0) runtime note:

- On Blackwell GPUs (RTX 50-series, RTX PRO 6000) with a CUDA toolkit older than 12.9, flashinfer's JIT sampler build targets `compute_120f`, which older `nvcc` rejects. The backend works around this automatically before importing vLLM by setting `VLLM_USE_FLASHINFER_SAMPLER=0` (falls back to the native PyTorch sampler) and adding the active venv's bin directory to `PATH` (so flashinfer's JIT can find `ninja`). Both are applied with `setdefault`, so you can override them. Installing CUDA toolkit 12.9+ removes the need for the sampler workaround.

## Remote Backends

Remote models use the same public model contract as local models, but activate an upstream API route instead of loading model weights. For V1, the supported remote backend is `openai_compatible` with Chat Completions.

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
        "remote_model": "provider-model",
        "remote_timeout_s": 120,
        "remote_health_check": "config_only",
        "remote_max_retries": 0,
        "remote_thinking": "disabled",
        "target_inflight": 1,
        "enabled": false
      }
    }
  }
}
```

Remote backend notes:

- `model_path` is not required for `openai_compatible` models.
- The API key is read from the environment variable named by `remote_api_key_env`.
- Callers must set `allow_remote: true`; otherwise the request is rejected before it enters the scheduler.
- `target_inflight` controls local submission concurrency. It is not a guarantee that the upstream provider runs requests concurrently.
- `remote_thinking` is an explicit provider extension for APIs that expose a Chat Completions `thinking` field. It currently accepts `"enabled"` or `"disabled"`.
- Remote calls may incur provider costs. Phase 1 only provides the explicit `allow_remote` request gate; local budget enforcement and cost ledgers are still design-note work. See [remote-openai-compatible-backend-notes.md](docs/remote-openai-compatible-backend-notes.md).

## Replicas

- Clients send only the model id that appears in the API and config.
- A single model id may map to multiple identical internal replicas.
- `/v1/models` returns model ids, not internal replica ids.
- `/v1/admin/models` returns one aggregate row per model id.
- `replicas` is the default replica count for that model id when it is loaded.
- `replica_max` is the maximum allowed replica count for that model id.
- Replicas are only for identical runtime instances. Different context sizes or cache settings require different model ids.

## Timing Metrics

The response `metrics` payload uses nested timers:

- `backend_inference_wall_ms`
  time spent inside the model runtime itself generating the result
- `engine_total_wall_ms`
  backend inference plus queueing, scheduling, and other engine work around it
- `pool_total_wall_ms`
  total time spent inside the `llm-pool` request handler

In other words:

- `Inference` is the smallest boundary
- `Engine` wraps `Inference`
- `Pool` wraps `Engine`

The payload may also include runtime-specific counters and sub-timers:

- `engine_queue_wait_ms`
  time spent waiting in the per-model scheduler queue before backend work starts
- `engine_tokenize_ms`
  prompt tokenization time when the backend reports it separately
- `gpu_time_to_first_token_ms`
  time from generation start to first generated token, when available
- `gpu_generate_total_ms`
  backend-reported generation time
- `gpu_decode_after_first_token_ms`
  generation time after the first token, when available
- `engine_prompt_tokens` / `engine_output_tokens`
  prompt and generated token counts, when available
- `engine_tokens_per_second`
  generated output tokens divided by the measured generation wall time

Some fields are backend-dependent and may be `null`.

## Test

```bash
python3 -m unittest discover -s tests
```

## Design Notes

The repo also includes design notes and trackers in various stages of completion:

- [runtime-scheduler-notes.md](docs/runtime-scheduler-notes.md)
  Captures the broader scheduler design space beyond the current in-process implementation.
- [runtime-scheduler-tracker.md](docs/runtime-scheduler-tracker.md)
  Tracks the current scheduler MVP implementation status and remaining deferred work.
- [model-replica-routing-notes.md](docs/model-replica-routing-notes.md)
  Captures the client-visible model and replica-routing semantics.
- [remote-openai-compatible-backend-notes.md](docs/remote-openai-compatible-backend-notes.md)
  Captures the proposed remote OpenAI-compatible backend shape, including cost-control notes.
- [runtime-subprocess-notes.md](docs/runtime-subprocess-notes.md)
  Captures the intended process-isolation model for loaded runtimes and how that should fit behind the same scheduler/runtime adapter boundary.
- [gemma4-mtp-vllm-notes.md](docs/gemma4-mtp-vllm-notes.md)
  Captures the vLLM backend and Gemma 4 MTP speculative-decoding path; the in-process vLLM backend is implemented, the MTP path is wired but not yet verified.

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
- vLLM

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
