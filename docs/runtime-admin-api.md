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
      "replicas": 3,
      "replica_max": 4,
      "loaded_replicas": 3,
      "inflight_requests": 0,
      "queue_depth": 0,
      "runtime_inflight": 0,
      "configured_target_inflight": 1,
      "effective_target_inflight": 1,
      "last_error": null,
      "vram_estimate_mib": 57200,
      "vram_estimate_replica_count": 3,
      "vram_estimate_source": "model_artifact_size",
      "load_constraints": {
        "gguf_n_ctx": {
          "kind": "integer",
          "minimum": 1,
          "step": 1
        },
        "gguf_flash_attn": {
          "kind": "enum",
          "default": "auto",
          "allowed_values": ["on", "off", "auto"],
          "examples": ["auto", "on", "off"]
        },
        "gguf_type_k": {
          "kind": "string_or_null",
          "format": "ggml_type_name",
          "default": "f16",
          "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
          "examples": ["f16", "q8_0", "q4_0"]
        },
        "gguf_type_v": {
          "kind": "string_or_null",
          "format": "ggml_type_name",
          "default": "f16",
          "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
          "examples": ["f16", "q8_0", "q4_0"]
        }
      },
      "load_recommendations": {
        "gguf_cache_type_pairs": {
          "kind": "pair_presets",
          "fields": ["gguf_type_k", "gguf_type_v"],
          "recommended_pairs": [
            {
              "label": "f16/f16",
              "gguf_type_k": "f16",
              "gguf_type_v": "f16"
            },
            {
              "label": "q8_0/q8_0",
              "gguf_type_k": "q8_0",
              "gguf_type_v": "q8_0"
            },
            {
              "label": "q4_0/q4_0",
              "gguf_type_k": "q4_0",
              "gguf_type_v": "q4_0"
            }
          ],
          "notes": [
            "Service-curated presets for GGUF cache types.",
            "Prefer symmetric GGUF K/V pairs by default; asymmetric pairs may reduce or disable GPU offload in upstream llama.cpp."
          ]
        }
      },
      "load_override": {},
      "capabilities": {
        "modalities": ["text"]
      },
      "definition": {
        "model_path": "/home/gunnar/models/google_gemma-4-E2B-it-Q8_0/google_gemma-4-E2B-it-Q8_0.gguf",
        "backend": "gguf",
        "prompt_format": "gemma4_template",
        "enabled": true,
        "replicas": 3,
        "replica_max": 4,
        "target_inflight": 1,
        "gguf_n_gpu_layers": -1,
        "gguf_n_ctx": 4096,
        "gguf_flash_attn": "auto",
        "gguf_type_k": null,
        "gguf_type_v": null
      }
    }
  ]
}
```

Notes:

- `configured_enabled` reports what the merged config says
- `runtime_state` reports the live process state
- `replicas` reports the current effective replica count for the admin row
- `definition.replicas` reports the configured default replica count
- `loaded_replicas` reports how many replicas of the public model are currently loaded
- `queue_depth` is the public-model queue depth inside the scheduler
- `runtime_inflight` is aggregate inflight work across loaded replicas of the public model
- `configured_target_inflight` is the configured per-replica inflight target
- `effective_target_inflight` is the currently honest per-replica scheduler target after capability clamping
- `vram_estimate_mib` is an approximate per-model VRAM estimate
- `vram_estimate_replica_count` is the replica count that the VRAM estimate was measured or derived for
- `vram_estimate_source` is either `observed_load_delta`, `model_artifact_size`, or `unavailable`
- `capabilities.modalities` lists which input modalities the model accepts (`["text"]` or `["text", "image"]`); a UI can use it to decide whether to allow image input for a model
- `load_constraints` describes backend-specific live-load fields for UI controls
- `load_recommendations` describes service-curated recommended presets and pairings for UI defaults
- `load_override` reports the runtime-only override currently active on a loaded model
- `definition` contains common model fields plus only the fields relevant to the resolved backend

#### UI-Facing `load_constraints`

For UI work, `load_constraints` is the source of truth for which live-load controls should be shown for a model.

Rules:

- if a field is absent from `load_constraints`, the UI should treat that field as unsupported for that model
- for `kind: "integer"`, the UI should use `minimum` and `step` directly for numeric inputs or sliders
- for `kind: "enum"`, the UI should use `allowed_values` directly for a constrained select or segmented control
- for `kind: "string_or_null"`, the UI should use a text input or a constrained select if the frontend chooses to offer known values
- if a `default` is present in `load_constraints`, the UI may use it as the concrete runtime default when both `definition` and `load_override` resolve to `null`
- `load_constraints` is derived from the resolved backend, not from whether the model is currently loaded or unloaded
- when this document and upstream backend docs differ, the UI should follow the live `load_constraints` payload returned by the service

#### UI-Facing `load_recommendations`

For UI work, `load_recommendations` is the source of truth for which presets the service recommends surfacing first.

Rules:

- `load_recommendations` is optional and additive; it does not replace `load_constraints`
- fields listed in `recommended_pairs` must still be validated against `load_constraints`
- the service may accept more combinations than it recommends
- the UI should treat these presets as convenience defaults, not as an exhaustive list of allowed values

#### Effective Loaded Values

For UI state, `definition` and `load_override` should be interpreted together:

- `definition` is the configured value from merged config
- `load_override` is a runtime-only sparse patch
- the effective loaded value is computed by applying `load_override` over `definition`
- key presence in `load_override` matters, even when the value is `null`

This means the UI should not use truthiness to merge values.

Correct merge rule:

```text
if key exists in load_override:
  effective_value = load_override[key]
else:
  effective_value = definition[key]
```

This matters in particular for `exllama_cache_quant`.

Example:

```json
{
  "load_override": {
    "exllama_cache_quant": null
  },
  "definition": {
    "exllama_cache_quant": "8,8"
  }
}
```

In this case, the effective loaded value is `null`.

For GGUF, the UI may interpret the effective cache type value as:

- field absent in `load_override` and absent or `null` in `definition`: use `load_constraints.<field>.default`, currently `"f16"`
- effective value `null`: use `load_constraints.<field>.default`, currently `"f16"`
- effective value `"q8_0"`: `q8_0`
- effective value `"q4_0"`: `q4_0`

For ExLlamaV3, the UI may interpret the effective quant value as:

- field absent in `load_override` and absent or `null` in `definition`: fp16
- effective value `null`: fp16
- effective value `"8"`: `k=8`, `v=8`
- effective value `"8,4"`: `k=8`, `v=4`

The API does not currently return separate `k_bits` and `v_bits` fields.
The UI should parse `exllama_cache_quant` itself when it wants to display separate K/V values.

Current shapes:

GGUF:

```json
{
  "gguf_n_ctx": {
    "kind": "integer",
    "minimum": 1,
    "step": 1
  },
  "gguf_flash_attn": {
    "kind": "enum",
    "default": "auto",
    "allowed_values": ["on", "off", "auto"],
    "examples": ["auto", "on", "off"]
  },
  "gguf_type_k": {
    "kind": "string_or_null",
    "format": "ggml_type_name",
    "default": "f16",
    "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
    "examples": ["f16", "q8_0", "q4_0"]
  },
  "gguf_type_v": {
    "kind": "string_or_null",
    "format": "ggml_type_name",
    "default": "f16",
    "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
    "examples": ["f16", "q8_0", "q4_0"]
  }
}
```

GGUF recommended presets:

```json
{
  "gguf_cache_type_pairs": {
    "kind": "pair_presets",
    "fields": ["gguf_type_k", "gguf_type_v"],
    "recommended_pairs": [
      {
        "label": "f16/f16",
        "gguf_type_k": "f16",
        "gguf_type_v": "f16"
      },
      {
        "label": "q8_0/q8_0",
        "gguf_type_k": "q8_0",
        "gguf_type_v": "q8_0"
      },
      {
        "label": "q4_0/q4_0",
        "gguf_type_k": "q4_0",
        "gguf_type_v": "q4_0"
      }
    ]
  }
}
```

ExLlamaV3:

```json
{
  "exllama_cache_size": {
    "kind": "integer",
    "minimum": 256,
    "step": 256
  },
  "exllama_max_rq_tokens": {
    "kind": "integer",
    "minimum": 1,
    "step": 1
  },
  "exllama_cache_k_bits": {
    "kind": "integer_or_null",
    "minimum": 2,
    "maximum": 8,
    "default": null,
    "null_means": "fp16",
    "allowed_values": [2, 3, 4, 5, 6, 7, 8]
  },
  "exllama_cache_v_bits": {
    "kind": "integer_or_null",
    "minimum": 2,
    "maximum": 8,
    "default": null,
    "null_means": "fp16",
    "allowed_values": [2, 3, 4, 5, 6, 7, 8]
  },
  "exllama_cache_quant": {
    "kind": "string_or_null",
    "format": "<bits>|<k_bits>,<v_bits>"
  }
}
```

vLLM:

```json
{
  "vllm_max_model_len": {
    "kind": "integer",
    "minimum": 256,
    "step": 256
  },
  "vllm_kv_cache_dtype": {
    "kind": "enum",
    "default": "auto",
    "allowed_values": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
    "examples": ["auto", "fp8"]
  },
  "vllm_kv_cache_memory_bytes": {
    "kind": "integer",
    "minimum": 268435456,
    "step": 268435456,
    "unit": "bytes",
    "display_unit": "mib"
  },
  "vllm_max_pixels": {
    "kind": "integer",
    "minimum": 200704,
    "step": 200704,
    "unit": "pixels"
  }
}
```

ExLlamaV3 recommended presets:

```json
{
  "exllama_cache_bit_pairs": {
    "kind": "pair_presets",
    "fields": ["exllama_cache_k_bits", "exllama_cache_v_bits"],
    "recommended_pairs": [
      {
        "label": "fp16",
        "exllama_cache_k_bits": null,
        "exllama_cache_v_bits": null
      },
      {
        "label": "8/8",
        "exllama_cache_k_bits": 8,
        "exllama_cache_v_bits": 8
      },
      {
        "label": "8/4",
        "exllama_cache_k_bits": 8,
        "exllama_cache_v_bits": 4
      }
    ]
  }
}
```

CT2 and stub:

```json
{}
```

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
      "configured_target_inflight": 1,
      "effective_target_inflight": 1,
      "vram_estimate_mib": 12500,
      "vram_estimate_replica_count": 3,
      "vram_estimate_source": "model_artifact_size"
    },
    {
      "name": "mistral-small-3.2-24b-instruct-2506-gguf",
      "runtime_state": "unloaded",
      "is_loaded": false,
      "configured_target_inflight": 1,
      "effective_target_inflight": 1,
      "vram_estimate_mib": 16800,
      "vram_estimate_replica_count": 1,
      "vram_estimate_source": "model_artifact_size"
    }
  ],
  "error": null
}
```

Notes:

- `used_over_total` matches the compact view you typically read from `nvidia-smi`
- `vram_estimate_mib` for unloaded models is still an estimate, not a reservation
- `vram_estimate_replica_count` tells the caller how many replicas that estimate corresponds to
- if `nvidia-smi` is unavailable, `gpus` can be empty and `error` will explain why

### `POST /v1/admin/models/{model_name}/load`

Loads one model that already exists in merged config.

Rules:

- `404` if `model_name` is unknown
- `200` if the model is already `loaded` or `loading`
- transition `unloaded -> loading -> loaded`
- transition `failed -> loading -> loaded`
- if load fails, transition to `failed` and retain `last_error`
- an optional request body may provide `replicas` for this load, but only while the model is `unloaded` or `failed`
- an optional request body may provide temporary backend-specific load overrides for this one live load

Supported load override fields:

- public model: `replicas`
- GGUF: `gguf_n_ctx`, `gguf_flash_attn`, `gguf_type_k`, `gguf_type_v`
- ExLlamaV3: `exllama_cache_size`, `exllama_cache_quant`, `exllama_cache_k_bits`, `exllama_cache_v_bits`, `exllama_max_rq_tokens`
- vLLM: `vllm_max_model_len`, `vllm_kv_cache_dtype`, `vllm_kv_cache_memory_bytes`, `vllm_max_pixels`

Example load bodies:

```json
{
  "replicas": 3
}
```

```json
{
  "gguf_n_ctx": 8192
}
```

```json
{
  "gguf_n_ctx": 16384
}
```

```json
{
  "gguf_n_ctx": 32768
}
```

```json
{
  "gguf_n_ctx": 32768,
  "gguf_flash_attn": "auto",
  "gguf_type_k": "q8_0",
  "gguf_type_v": "q4_0"
}
```

```json
{
  "exllama_cache_size": 32768,
  "exllama_cache_quant": null,
  "exllama_max_rq_tokens": 32768
}
```

```json
{
  "exllama_cache_size": 32768,
  "exllama_cache_quant": "8,8",
  "exllama_max_rq_tokens": 32768
}
```

```json
{
  "exllama_cache_size": 32768,
  "exllama_cache_quant": "8,4",
  "exllama_max_rq_tokens": 32768
}
```

```json
{
  "exllama_cache_size": 32768,
  "exllama_cache_k_bits": 8,
  "exllama_cache_v_bits": 4,
  "exllama_max_rq_tokens": 32768
}
```

```json
{
  "vllm_max_model_len": 16384,
  "vllm_kv_cache_dtype": "fp8",
  "vllm_kv_cache_memory_bytes": 2147483648,
  "vllm_max_pixels": 4014080
}
```

vLLM load override notes:

- `vllm_max_model_len` is the per-load context length.
- `vllm_kv_cache_dtype` quantizes the KV cache; allowed UI values are `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`. The service accepts any dtype string vLLM supports.
- `vllm_kv_cache_memory_bytes` sets an absolute KV cache size in bytes. It is machine-independent and overrides `vllm_gpu_memory_utilization` for KV sizing. The `load_constraints` entry carries `unit: "bytes"` and `display_unit: "mib"` so the UI can present it in MiB.
- `vllm_max_pixels` caps the vision-token budget per image for vision-language models; it is merged into the model's `vllm_mm_processor_kwargs` as `max_pixels`.

`exllama_cache_quant` format:

- omitted or `null`: fp16 KV cache
- `"<bits>"`: same quantization for K and V, for example `"8"`
- `"<k_bits>,<v_bits>"`: separate K/V quantization, for example `"8,4"`

`exllama_cache_k_bits` and `exllama_cache_v_bits` format:

- both omitted: do not override the current configured value
- both `null`: reset to fp16 KV cache
- both integers from `2` through `8`: override K and V separately
- they must be provided together
- they cannot be combined with `exllama_cache_quant` in the same load request

`gguf_type_k` and `gguf_type_v` format:

- omitted or `null`: use the runtime default cache type
- `"<ggml_type_name>"`: a GGML cache type name, for example `"f16"`, `"q8_0"`, or `"q4_0"`

`gguf_flash_attn` format:

- omitted: do not override the current configured value
- `"on"`: force Flash Attention on
- `"off"`: force Flash Attention off
- `"auto"`: use the runtime auto mode

Upstream references for these backend-specific value sets:

- GGUF `allowed_values` and default `f16` are based on the official `llama.cpp` server docs:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- ExLlamaV3 `k_bits`/`v_bits` allowed range `2..8` is based on the official ExLlamaV3 README, which documents `2-8 bit cache quantization`:
  https://github.com/turboderp-org/exllamav3
- The conservative GGUF preset guidance above is informed by upstream llama.cpp reports that asymmetric K/V pairs can disable GPU offload in some setups:
  https://github.com/ggml-org/llama.cpp/issues/20866

These overrides are runtime-only:

- they do not modify `settings.json`
- they do not modify `local.json`
- they should be surfaced separately from the configured definition in admin responses
- `replicas` in the load request does not modify `definition.replicas`; it only selects the replica count for that one live load

Suggested response shape:

```json
{
  "name": "google_gemma-4-E2B-it-Q8_0-gguf",
  "resolved_backend": "gguf",
  "configured_enabled": false,
  "runtime_state": "loaded",
  "is_loaded": true,
  "replicas": 3,
  "replica_max": 4,
  "loaded_replicas": 3,
  "inflight_requests": 0,
  "queue_depth": 0,
  "runtime_inflight": 0,
  "configured_target_inflight": 1,
  "effective_target_inflight": 1,
  "last_error": null,
  "vram_estimate_mib": 12340,
  "vram_estimate_replica_count": 3,
  "vram_estimate_source": "observed_load_delta",
  "load_constraints": {
    "gguf_n_ctx": {
      "kind": "integer",
      "minimum": 1,
      "step": 1
    },
    "gguf_type_k": {
      "kind": "string_or_null",
      "format": "ggml_type_name",
      "default": "f16",
      "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
      "examples": ["f16", "q8_0", "q4_0"]
    },
    "gguf_type_v": {
      "kind": "string_or_null",
      "format": "ggml_type_name",
      "default": "f16",
      "allowed_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
      "examples": ["f16", "q8_0", "q4_0"]
    }
  },
  "load_recommendations": {
    "gguf_cache_type_pairs": {
      "kind": "pair_presets",
      "fields": ["gguf_type_k", "gguf_type_v"],
      "recommended_pairs": [
        {
          "label": "f16/f16",
          "gguf_type_k": "f16",
          "gguf_type_v": "f16"
        },
        {
          "label": "q8_0/q8_0",
          "gguf_type_k": "q8_0",
          "gguf_type_v": "q8_0"
        },
        {
          "label": "q4_0/q4_0",
          "gguf_type_k": "q4_0",
          "gguf_type_v": "q4_0"
        }
      ]
    }
  },
  "load_override": {
    "gguf_n_ctx": 32768,
    "gguf_flash_attn": "auto",
    "gguf_type_k": "q8_0",
    "gguf_type_v": "q4_0"
  },
  "definition": {
    "model_path": "/home/gunnar/models/google_gemma-4-E2B-it-Q8_0/google_gemma-4-E2B-it-Q8_0.gguf",
    "backend": "gguf",
    "prompt_format": "gemma4_template",
    "enabled": false,
    "replicas": 3,
    "replica_max": 4,
    "target_inflight": 1,
    "gguf_n_gpu_layers": -1,
    "gguf_n_ctx": 4096,
    "gguf_flash_attn": "auto",
    "gguf_type_k": null,
    "gguf_type_v": null
  }
}
```

Notes:

- loading is allowed for configured models even when `configured_enabled` is `false`
- after a successful load, `vram_estimate_source` may switch to `observed_load_delta` if a GPU delta could be measured during load
- when `replicas` is provided, the model load is aggregate and all-or-nothing for that selected replica count
- `replicas` and load overrides may only be changed while the model is `unloaded` or `failed`

Validation behavior:

- `422` means the request body failed schema validation before runtime logic ran
  Examples:
  `gguf_n_ctx: 0`
  `exllama_cache_size: 0`
  `exllama_max_rq_tokens: 0`
- `400` with `code: "invalid_load_request"` means the body was structurally valid, but the values were invalid for the resolved backend or runtime rules
  Examples:
  `gguf_type_k: "q8-0"`
  `gguf_type_k: "foo"`
  sending only `exllama_cache_k_bits` without `exllama_cache_v_bits`
  combining `exllama_cache_quant` with `exllama_cache_k_bits`/`exllama_cache_v_bits`
  `exllama_cache_size: 8000`
  `exllama_cache_quant: "fp16"`
  sending ExLlamaV3-only fields to a GGUF model
  sending load overrides while the model is already loaded and not first unloading it
- `409` still applies for runtime state conflicts such as loading or unloading transitions

### `POST /v1/admin/models/{model_name}/unload`

Gracefully unloads one currently loaded model.

Rules:

- `404` if `model_name` is unknown
- `200` if already `unloaded`
- `200` if already `unloading`
- transition `loaded -> unloading -> unloaded`
- unload stops all loaded replicas of the public model
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
  "replicas": 3,
  "replica_max": 4,
  "loaded_replicas": 0,
  "inflight_requests": 0,
  "queue_depth": 0,
  "runtime_inflight": 0,
  "configured_target_inflight": 1,
  "effective_target_inflight": 1,
  "last_error": null,
  "vram_estimate_mib": 12340,
  "vram_estimate_replica_count": 4,
  "vram_estimate_source": "observed_load_delta",
  "definition": {
    "model_path": "/home/gunnar/models/google_gemma-4-E2B-it-Q8_0/google_gemma-4-E2B-it-Q8_0.gguf",
    "backend": "gguf",
    "device": "cuda",
    "prompt_format": "gemma4_template",
    "enabled": false,
    "replicas": 3,
    "replica_max": 4,
    "target_inflight": 1
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
- retry policy for failed model loads
