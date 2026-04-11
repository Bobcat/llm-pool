# llm-pool-dev

Small FastAPI service that exposes a generic `POST /v1/responses` contract for local LLM inference.

Current scope:
- generic request schema with per-request decoding params
- JSON response mode
- SSE streaming mode
- CT2 and ExLlamaV3 engines behind the same API contract
- per-model backend routing inside one llm-pool instance
- `config/settings.json` and `deploy/systemd/` repo scaffolding

## Endpoint

`POST /v1/responses`

Example request:

```json
{
  "model": "eurollm-9b-ct2-int8",
  "input": "Hello world",
  "instructions": "Translate to Dutch.",
  "stream": true,
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

## Run

The default config in `config/settings.json` points at the local EuroLLM CT2 model directory.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8011
```

## Local Overrides

You can keep shared defaults in `config/settings.json` and put machine-local overrides in `config/local.json`.
When present, `local.json` is merged over `settings.json` (override wins per key).

Per model, you can optionally override the backend:

```json
{
  "engine": {
    "backend": "ct2",
    "models": {
      "eurollm-9b-ct2-int8": {
        "model_path": "/models/eurollm-ct2",
        "backend": "ct2"
      },
      "gemma-4-26B-A4B-it-exl3-5.10bpw": {
        "model_path": "/home/gunnar/models/gemma-4-26B-A4B-it-exl3-5.10bpw",
        "backend": "exllamav3",
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
- ExLlamaV3 dependencies are loaded lazily and required only when an ExLlamaV3 model is configured.

Optional env vars:
- `LLM_POOL_SETTINGS_PATH`: explicit base settings file path.
- `LLM_POOL_LOCAL_SETTINGS_PATH`: explicit local override file path.

## Test

```bash
python3 -m unittest discover -s tests
```
