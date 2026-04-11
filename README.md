# llm-pool-dev

Small FastAPI service that exposes a generic `POST /v1/responses` contract for local LLM inference.

Current scope:
- generic request schema with per-request instructions and decoding params
- JSON response mode
- SSE streaming mode
- `GET /v1/models` endpoint for enabled model discovery
- per-request runtime metrics in the response
- CT2 and ExLlamaV3 engines behind the same API contract
- per-model backend routing inside one llm-pool instance
- includes model serving and routing, but no queue/scheduler layer yet (unlike asr-pool)

## Endpoint

`POST /v1/responses`

- `stream: false` returns a JSON response envelope.
- `stream: true` returns SSE events: `response.created`, `response.output_text.delta`, `response.metrics`, `response.completed`.

`GET /v1/models`

- returns the current `default_model` and the enabled loaded models.

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
    "max_tokens": 256,
    "stop": ["<|im_end|>"]
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

## Local Overrides

You can keep shared defaults in `config/settings.json` and put machine-local overrides in `config/local.json`.
When present, `local.json` is merged over `settings.json` (override wins per key).

Settings files can also define `service.host`, `service.port`, `service.log_level`, `engine.default_model`, and global `engine.decoding` defaults.

Per model, you can set `model_path`, `device`, `compute_type`, `prompt_format`, `enabled`, and optionally override the backend:

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
        "enabled": true,
        "exllama_cache_size": 16384,
        "exllama_cache_quant": "8,8",
        "exllama_tensor_parallel": true,
        "exllama_gpu_split": "24,24"
      }
    }
  }
}
```

Notes:
- Models without a `backend` field use the global `engine.backend`.
- `enabled` controls whether a model is loaded by the pool at startup.
- Request-level decoding values override `engine.decoding` defaults when provided.
- ExLlamaV3 models also support `exllama_tp_backend`, `exllama_max_batch_size`, `exllama_max_chunk_size`, `exllama_max_q_size`, and `exllama_max_rq_tokens`.
- ExLlamaV3 dependencies are loaded lazily and required only when an ExLlamaV3 model is configured.

Optional env vars:
- `LLM_POOL_SETTINGS_PATH`: explicit base settings file path.
- `LLM_POOL_LOCAL_SETTINGS_PATH`: explicit local override file path.

## Test

```bash
python3 -m unittest discover -s tests
```
