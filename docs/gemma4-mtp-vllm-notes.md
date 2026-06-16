# Gemma 4 MTP vLLM Backend Notes

This note captures a possible Gemma 4 MTP backend path for `llm-pool`.

It is a design note, not an implementation spec.

Status: implemented for the managed `vllm_serve` path; partially implemented for the in-process `vllm` path.

Current reality note:

- The `backend: "vllm"` adapter described below now exists (`app/engine/vllm.py`).
- It runs the vLLM Python engine in-process via `AsyncLLMEngine.from_engine_args`,
  on a dedicated event-loop thread. It does not start `vllm serve` and does not
  call vLLM over the OpenAI-compatible HTTP API, matching the design here.
- A separate `backend: "vllm_serve"` now exists for models that should run
  through upstream `vllm serve` as a managed local subprocess. It reuses the
  same `vllm_*` config fields for target model, multimodal settings, and
  speculative/MTP config, while `vllm_serve_*` fields control the subprocess.
- The speculative-decoding config surface from the "Config Shape" section is
  wired: `vllm_speculative_method`, `vllm_speculative_model`, and
  `vllm_num_speculative_tokens` are passed through to vLLM's `speculative_config`
  for the in-process backend and to `--speculative-config` for `vllm_serve`.
- Gemma 4 26B A4B NVFP4 plus the official Gemma 4 assistant checkpoint has
  been run end to end through `vllm_serve` with image input. The local
  document-structure OCR benchmark currently uses
  `vllm_num_speculative_tokens: 8`; the README records the observed sweep.
- The `vllm_serve` Gemma 4 NVFP4 model also uses `vllm_serve_extra_args`
  with `--max-num-seqs 1` for single-user Workbench use, because vLLM's
  broader default scheduler/CUDA-graph capture sizes can reserve much more
  VRAM than the explicit KV-cache budget suggests.
- Beyond this note's scope, the same backend also gained multimodal (image)
  input support, used by the image-description / OCR-grounding work.
- A separate `backend: "llama_server"` now exists for GGUF models that need
  native llama-server features such as `--mmproj` and MTP draft flags. That is
  a separate local backend path and does not change this note's vLLM-specific
  architecture.
- **Not yet done / verified:** actual Gemma 4 MTP has not been run end to end
  through the in-process `vllm` backend. The speculative path is wired but
  untested there against real Gemma 4 target and assistant checkpoints.
  `llm-pool` does not yet expose MTP-specific acceptance-rate metrics in its
  own response/admin payloads, and the generic `llm-pool` runtime subprocess
  architecture is not implemented.
- Blackwell (SM 12.0) hosts with a CUDA toolkit < 12.9 need a small runtime
  workaround that the backend applies automatically; see the README's vLLM
  backend notes.

## Purpose

Gemma 4 exposes Multi-Token Prediction speculative decoding through small assistant checkpoints.

The goal is to support that path without pretending it is just another ordinary small model call inside `llm-pool`.

The intended first target is:

- vLLM as the inference backend
- official Hugging Face Gemma 4 target checkpoints
- official Hugging Face Gemma 4 assistant checkpoints
- vLLM's `method: "mtp"` speculative decoding configuration

## Main Architectural Decision

Gemma 4 MTP should be treated as a backend capability, not as router-level orchestration.

The `llm-pool` router should still see one public model id and one backend runtime. The target model and assistant model should be loaded and coordinated by vLLM.

This matters because Gemma 4 assistant checkpoints are not generic draft models in vLLM. vLLM maps Gemma 4 assistant checkpoints to its Gemma 4 MTP implementation and wires the assistant to the target runtime.

## Backend Shape

The clean first backend shape is a new backend adapter:

- `backend: "vllm"`

The backend should represent one vLLM-served model route.

V1 should use the vLLM Python engine in-process.

The intended runtime shape is:

```text
llm-pool backend runtime
  -> in-process vLLM Python engine
```

The backend should not start `vllm serve` and then call it through the OpenAI-compatible HTTP API. That would add a second serving process, a second queue/scheduler layer, and another lifecycle surface underneath `llm-pool`.

The likely vLLM API target is `AsyncLLMEngine` / `AsyncLLM`, because vLLM documents the simpler `LLM` entrypoint as an offline inference API. `LLM` may still be useful for a local spike, but it should not be assumed to be the final backend API.

## Runtime Boundary

This note intentionally keeps vLLM inside the `llm-pool` backend runtime.

If runtime subprocess isolation is introduced later, the intended shape becomes:

```text
llm-pool parent process
  -> llm-pool runtime child process
       -> in-process vLLM Python engine
```

It should not become:

```text
llm-pool parent process
  -> llm-pool runtime child process
       -> vLLM server process
            -> vLLM engine
```

The subprocess boundary, if added, should belong to `llm-pool`, not to a nested vLLM server deployment.

See `runtime-subprocess-notes.md`.

## Config Shape

V1 config should keep the target model and MTP assistant explicit.

Example:

```json
{
  "engine": {
    "models": {
      "gemma4-e2b-mtp": {
        "backend": "vllm",
        "model_path": null,
        "prompt_format": "gemma4_template",
        "vllm_model": "google/gemma-4-E2B-it",
        "vllm_tensor_parallel_size": 1,
        "vllm_max_model_len": 8192,
        "vllm_speculative_method": "mtp",
        "vllm_speculative_model": "google/gemma-4-E2B-it-assistant",
        "vllm_num_speculative_tokens": 1,
        "enabled": false
      }
    }
  }
}
```

`vllm_model` is the target model id or local path. `model_path` is not required for vLLM models and may be `null`.

The assistant model should not be hidden in a generic `small_model` field. It is part of the backend's speculative decoding configuration.

## GGUF Position

GGUF should not be the first Gemma 4 MTP target for the vLLM path.

vLLM has GGUF loading support, but vLLM documents it as experimental and under-optimized. For Gemma 4 MTP, the first path should use official Hugging Face target and assistant checkpoints.

The existing `gguf` backend should remain the llama.cpp path.

If llama.cpp later supports official Gemma 4 MTP well enough through GGUF, that should be handled as a separate enhancement to the existing `gguf` backend, not as part of the vLLM backend.

## Request Behavior

Clients should still send the public `llm-pool` model id:

```json
{
  "model": "gemma4-e2b-mtp",
  "input": "Write a small Python function."
}
```

The client should not send:

- the assistant model id
- the speculative method
- arbitrary vLLM runtime arguments

Those are server-side runtime config.

## Metrics

V1 can preserve the current `EngineResult` shape and record ordinary prompt/output timing.

MTP-specific counters can be added later if exposed cleanly by vLLM, for example:

- draft acceptance rate
- proposed speculative tokens
- accepted speculative tokens

Those metrics are useful but not required for first functionality.

## Non-Goals

This in-process `backend: "vllm"` design does not introduce:

- a hand-written MTP decode loop inside `llm-pool`
- router-level coordination between two normal model calls
- a `vllm serve` subprocess managed by this backend; that now lives in the
  separate `backend: "vllm_serve"` path
- calls to vLLM through the OpenAI-compatible HTTP API
- support for an already-running external vLLM server in this backend
- generic request-time selection of arbitrary assistant models
- vLLM GGUF support as the initial Gemma 4 MTP path
- support for every vLLM speculative decoding method in V1
- automatic discovery of compatible Gemma 4 assistant checkpoints

## Open Questions

- Should the backend use `AsyncLLMEngine` or the newer `AsyncLLM` alias directly?
- Which additional vLLM runtime settings should be first-class config fields versus opaque extra args?
- Should MTP-specific metrics be pulled from vLLM logs, vLLM metrics, or omitted in V1?

## References

- vLLM MTP docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- vLLM speculative decoding docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM GGUF docs: https://docs.vllm.ai/en/stable/features/quantization/gguf/
- Gemma 4 E2B assistant checkpoint: https://huggingface.co/google/gemma-4-E2B-it-assistant
