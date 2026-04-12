# GGUF Backend Support for LLM Pool - Implementation Plan

## Background

`llm-pool-dev` is a FastAPI service that serves multiple LLM backends in-process behind one API contract.

Current backends:

- `Ct2Engine` for CT2 models
- `ExLlamaV3Engine` for EXL3 models

This plan adds a third backend:

- `LlamaCppEngine` for GGUF models, using `llama-cpp-python`

The goal of this plan is to add GGUF support in the same architectural style as the existing engines, without changing the public API contract.

## Decisions Already Made

- The GGUF backend uses `llama-cpp-python`.
- We use the mid-level API: `Llama.tokenize()`, `Llama.generate()`, and `Llama.detokenize()`.
- We do not use the high-level completion/chat APIs.
- The backend key is `gguf`.
- The runtime dataclass is `LlamaCppModelRuntime`.
- The engine class is `LlamaCppEngine`.

## CUDA Wheel Positioning

The current default assumption is that the CUDA-enabled `llama-cpp-python` wheel is a reasonable first path for NVIDIA support, but not something we should automatically assume is maximally tuned for a newest-generation GPU.

Practical interpretation:

- the CUDA wheel is an official and valid way to get GPU acceleration
- it is suitable for initial backend integration work
- it should be treated as a convenience build first, not as a guaranteed best-performance build for Blackwell-class hardware

Why this matters:

- `llama.cpp` itself is actively optimized for NVIDIA GPUs and has a serious CUDA backend
- but the wheel packaging/docs do not give a strong guarantee that the published CUDA wheel is built with the best possible tuning for every new GPU generation
- community reports suggest that source builds are still sometimes needed to get the best behavior or performance on newer NVIDIA cards

So for this GGUF backend plan, the working assumption should be:

- use the CUDA wheel as the default install path for bringing up the backend
- do not treat the wheel as proof of optimal performance
- if GPU behavior or performance looks suspicious on a Blackwell-class machine, be ready to validate against a source build later

## Scope

This plan covers:

- loading GGUF models through `llama-cpp-python`
- adding a new runtime and engine in `app/engine.py`
- extending `ModelSettings` with GGUF-specific settings
- routing GGUF models through the existing backend selection path
- unit tests for the GGUF engine

This plan does not cover:

- model-specific template selection
- deciding which models should use which `prompt_format`
- queue/scheduler work
- streaming changes
- backend performance tuning beyond basic GGUF settings

## Prompting Boundary

Prompt formatting remains a pool-level concern, not a GGUF-model selection concern.

For this backend we should keep the same existing contract:

- `prompt_format` continues to come from `ModelSettings`
- the GGUF backend should honor the same pool-level prompting rules as other backends
- this document does not prescribe which models should use which `prompt_format`

That keeps backend integration separate from model-level prompt decisions.

## Files Involved

- `app/config.py`
- `app/engine.py`
- `tests/test_engine.py`

## Configuration Changes

### `app/config.py`

Extend `ModelSettings` with a minimal set of GGUF-specific fields:

```python
gguf_n_gpu_layers: int = -1
gguf_n_ctx: int = 4096
gguf_flash_attn: bool = True
```

Meaning:

- `gguf_n_gpu_layers`: number of layers to offload to GPU, with `-1` meaning fully offloaded where possible
- `gguf_n_ctx`: context window size used when initializing the backend
- `gguf_flash_attn`: whether to enable flash attention when supported by the backend build

Then parse those fields in `load_settings()` in the same style as the existing `exllama_*` fields.

The GGUF settings should stay model-scoped, just like the ExLlamaV3 settings.

## Runtime Design

### `LlamaCppModelRuntime`

Add a new runtime dataclass in `app/engine.py`:

```python
@dataclass
class LlamaCppModelRuntime:
    config: ModelSettings
    llm: object
    generation_lock: threading.Lock = field(default_factory=threading.Lock)
```

Why this shape:

- `config` matches the existing runtime pattern
- `llm` is the loaded `llama_cpp.Llama` instance
- `generation_lock` serializes access to a loaded GGUF runtime, matching the current ExLlamaV3 approach

This keeps the runtime intentionally small. GGUF-specific state should live inside the `llama_cpp.Llama` instance unless we have a concrete reason to expose more.

## Engine Design

### `LlamaCppEngine`

Add a new engine class in `app/engine.py` following the same structure as `Ct2Engine` and `ExLlamaV3Engine`:

- read `default_model`
- read `decoding_defaults`
- build `_models`
- skip disabled models
- log and continue on model load failure
- fall back to the first loaded model if the configured default could not be loaded

This keeps backend startup behavior aligned across the codebase.

### `_build_runtime()`

Load the backend lazily:

```python
def _build_runtime(self, settings: ModelSettings) -> LlamaCppModelRuntime:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is required for the GGUF engine") from exc

    llm = Llama(
        model_path=settings.model_path,
        n_gpu_layers=settings.gguf_n_gpu_layers,
        n_ctx=settings.gguf_n_ctx,
        flash_attn=settings.gguf_flash_attn,
        verbose=False,
    )
    return LlamaCppModelRuntime(config=settings, llm=llm)
```

Notes:

- the import should remain inside `_build_runtime()`
- this keeps GGUF dependencies optional unless a GGUF model is actually configured
- the runtime should not eagerly add more wrapper state unless implementation work proves it necessary

### `complete()`

The GGUF `complete()` flow should mirror the current engine contract:

1. Resolve the runtime from `request.model`
2. Resolve the effective decoding parameters
3. Build the prompt text from the existing pool prompt-format contract
4. Tokenize with `llm.tokenize()`
5. Generate tokens with `llm.generate()`
6. Detokenize with `llm.detokenize()`
7. Return `EngineResult` with the standard metrics payload

Expected behavior:

- if `beam_size` is explicitly requested, log that GGUF ignores it and continue
- record prompt token count
- record time to first token
- record total generation time
- record output token count
- compute `engine_tokens_per_second`

### Stop Behavior

GGUF should follow the same user-visible stop behavior as the other backends as closely as possible.

That means generation should stop on:

- native EOS
- `max_tokens`
- configured stop strings

Important:

- stop strings should not be handled only by trimming the final text after generation
- the implementation should stop generation when a configured stop condition is reached

The exact mechanism can be decided during implementation:

- `stopping_criteria` passed to `llm.generate()`
- or an equivalent incremental stop check in the token loop

But the behavior should be real termination, not only post-processing.

## Prompt Rendering

This plan intentionally stays template-agnostic.

The only backend requirement is:

- GGUF must work with the existing `prompt_format` contract already used by the pool

This document does not define:

- which models should use `generic`
- which models should use any other prompt format

Those remain model/config decisions outside this backend plan.

If prompt rendering duplication becomes awkward during implementation, a small shared helper can be introduced later. That is optional and not required for the first GGUF backend patch.

## Decoding Defaults

Reuse the same `_resolve_decoding()` pattern already used by CT2 and ExLlamaV3.

That keeps request-level override behavior consistent across all backends:

- request value wins when provided
- otherwise engine defaults apply

## Routing Changes

### `ModelRouterEngine`

Extend `_build_backend_engine()` to support:

```python
if backend == "gguf":
    return LlamaCppEngine(settings)
```

### `build_engine()`

Extend direct backend construction to support:

```python
if settings.engine.backend == "gguf":
    return LlamaCppEngine(settings)
```

That is enough for both:

- global backend selection
- per-model backend overrides through `ModelRouterEngine`

## Test Plan

Add GGUF-focused unit tests in `tests/test_engine.py` using the same style as the current engine tests:

- fake backend objects
- `Engine.__new__()` to bypass heavy initialization
- manual `_models` setup

Recommended tests:

1. `test_llamacpp_complete_basic_flow`
   Verifies tokenization, generation, detokenization, and metrics wiring.

2. `test_llamacpp_complete_stops_on_eos`
   Verifies that generation stops when the backend yields EOS.

3. `test_llamacpp_complete_stops_on_max_tokens`
   Verifies that generation stops at the request/default max token limit.

4. `test_llamacpp_complete_logs_ignored_beam_size`
   Verifies that GGUF logs and ignores explicit `beam_size`, matching the ExLlamaV3 behavior pattern.

5. `test_model_router_dispatches_gguf_backend`
   Verifies that per-model backend routing correctly dispatches GGUF models to `LlamaCppEngine`.

6. `test_load_settings_parses_gguf_fields`
   Verifies that `gguf_n_gpu_layers`, `gguf_n_ctx`, and `gguf_flash_attn` are parsed into `ModelSettings`.

The tests should focus on backend contract behavior, not on model-specific prompt choices.

## Example Configuration

Example GGUF model entry in `config/local.json`:

```json
{
  "engine": {
    "models": {
      "example-gguf-model": {
        "model_path": "/home/gunnar/models/example.gguf",
        "backend": "gguf",
        "prompt_format": "generic",
        "enabled": true,
        "gguf_n_gpu_layers": -1,
        "gguf_n_ctx": 4096,
        "gguf_flash_attn": true
      }
    }
  }
}
```

The `prompt_format` in this example is only illustrative. Actual model-to-template selection remains a model/config concern.

## Verification

### Automated

Run the existing unit test suite, including the new GGUF tests:

```bash
python3 -m unittest discover -s tests -v
```

### Manual End-to-End

Assuming:

- `llama-cpp-python` is installed
- a GGUF model exists on disk
- `config/local.json` contains a GGUF model entry

Start the service and send a normal `/v1/responses` request.

Verify:

- the GGUF model loads successfully
- the response contains generated text
- `metrics.engine_prompt_tokens` is populated
- `metrics.engine_output_tokens` is populated
- `metrics.gpu_generate_total_ms` is populated
- `metrics.engine_tokens_per_second` is populated

## Implementation Notes

- `llm.tokenize()` expects `bytes`, not `str`
- `llm.detokenize()` returns `bytes`
- generation access should be serialized per runtime with `generation_lock`
- GGUF should ignore `beam_size` in a deliberate and logged way
- the backend should stop on EOS, `max_tokens`, and configured stop conditions

## Summary

The intended v1 change is small and honest:

- add a GGUF backend using `llama-cpp-python`
- keep the same API contract
- keep routing behavior consistent with the existing engines
- keep prompt-format selection out of backend design scope
- cover the new backend with focused unit tests
