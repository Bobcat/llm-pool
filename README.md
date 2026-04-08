# llm-pool-dev

Small FastAPI service that exposes a generic `POST /v1/responses` contract for local LLM inference.

Current scope:
- generic request schema with per-request decoding params
- JSON response mode
- SSE streaming mode
- CT2 engine behind the same API contract
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

## Test

```bash
python3 -m unittest discover -s tests
```
