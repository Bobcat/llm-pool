# Gemma 4 MTP vLLM Backend Notes

This note captures a possible Gemma 4 MTP backend path for `llm-pool`.

It is a design note, not an implementation spec.

Status: proposed.

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

Possible V1 implementation approaches:

- spawn/manage a local `vllm serve` subprocess and call its OpenAI-compatible API
- connect to an already-running vLLM OpenAI-compatible server

The subprocess option is more self-contained. The external-server option is smaller and overlaps with the remote OpenAI-compatible backend idea.

The decision does not need to be made in this note.

## Config Shape

V1 config should keep the target model and MTP assistant explicit.

Example:

```json
{
  "engine": {
    "models": {
      "gemma4-e2b-mtp": {
        "backend": "vllm",
        "model_path": "google/gemma-4-E2B-it",
        "prompt_format": "gemma4_template",
        "vllm_served_model_name": "gemma4-e2b-mtp",
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

`model_path` can remain the target model id for consistency with existing local backends, but for vLLM it may be a Hugging Face model id rather than a filesystem path.

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

This design does not introduce:

- a hand-written MTP decode loop inside `llm-pool`
- router-level coordination between two normal model calls
- generic request-time selection of arbitrary assistant models
- vLLM GGUF support as the initial Gemma 4 MTP path
- support for every vLLM speculative decoding method in V1
- automatic discovery of compatible Gemma 4 assistant checkpoints

## Open Questions

- Should V1 manage a local vLLM subprocess, or only call an already-running vLLM server?
- Should `model_path` be reused for vLLM model ids, or should vLLM get a dedicated `vllm_model` field?
- Which vLLM runtime settings should be first-class config fields versus opaque extra args?
- Should MTP-specific metrics be pulled from vLLM logs, vLLM metrics, or omitted in V1?

## References

- vLLM MTP docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- vLLM speculative decoding docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM GGUF docs: https://docs.vllm.ai/en/stable/features/quantization/gguf/
- Gemma 4 E2B assistant checkpoint: https://huggingface.co/google/gemma-4-E2B-it-assistant
