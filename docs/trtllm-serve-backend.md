# TensorRT-LLM Serve Backend Notes

The `trtllm_serve` backend runs NVIDIA TensorRT-LLM as a managed local server. It keeps TensorRT-LLM, CUDA, OpenMPI, and JIT compiler dependencies outside the llm-pool Python environment.

This note describes the backend as validated on 2026-09-01. TensorRT-LLM support changes quickly, so version-specific limitations below should be retested after an upgrade.

## Runtime Contract

Loading a model starts one command with this shape:

```text
trtllm-serve serve MODEL \
  --host HOST \
  --port PORT \
  --served_model_name MODEL_ALIAS \
  --no-telemetry \
  --config CONFIG.yaml
```

The backend then:

- waits for TensorRT-LLM's `/health` endpoint;
- sends non-streaming requests to `/v1/chat/completions`;
- forwards `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_tokens`, and `stop`;
- supports text and configured image input;
- supports multi-turn messages;
- reports text-only response-format support;
- runs the child in a separate process group;
- terminates the complete process group on unload or failed startup.

The public llm-pool SSE route remains available, but this adapter does not consume native token-streaming responses from TensorRT-LLM.

## Model Settings

| Field | Effect |
| --- | --- |
| `trtllm_model` | Model id, local checkpoint path, or TensorRT engine path passed as `MODEL`. Falls back to `model_path` when omitted. |
| `trtllm_trust_remote_code` | Adds `--trust_remote_code`. |
| `trtllm_serve_binary` | `trtllm-serve` executable. Its directory is prepended to `PATH`. |
| `trtllm_serve_host` | Local server bind host. |
| `trtllm_serve_port` | Fixed local port. When omitted, llm-pool selects a currently free port. |
| `trtllm_serve_model_alias` | Model name exposed to the upstream Chat Completions endpoint. |
| `trtllm_serve_timeout_s` | Timeout for one upstream inference request. |
| `trtllm_serve_start_timeout_s` | Maximum readiness wait during model load. |
| `trtllm_serve_stop_timeout_s` | SIGTERM grace period before process-group escalation. |
| `trtllm_serve_library_path` | Directories prepended to the child `LD_LIBRARY_PATH`. |
| `trtllm_serve_env` | Environment variables added to the child process. |
| `trtllm_serve_config_path` | YAML path passed to TensorRT-LLM as `--config`. |
| `trtllm_serve_reasoning_parser` | Adds TensorRT-LLM's `--reasoning_parser`. |
| `trtllm_serve_tool_parser` | Adds TensorRT-LLM's `--tool_parser`. This does not add tool-calling fields to llm-pool's public request schema. |
| `trtllm_serve_extra_args` | Additional `trtllm-serve serve` CLI arguments. |

These fields are part of the model definition. The runtime load API does not override them. The common `replicas` override remains available.

## TensorRT-LLM YAML

`trtllm_serve_config_path` is passed directly to the child process. llm-pool does not parse or merge the YAML. TensorRT-LLM documents `--config` as an alias of `--extra_llm_api_options`; explicit CLI flags win over YAML values.

The checked-in [Gemma 4 profile](../config/trtllm/gemma-4-26b-a4b-nvfp4.yaml) contains:

```yaml
max_batch_size: 4
max_num_tokens: 8192
max_seq_len: 20480
enable_chunked_prefill: false
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.5
```

The values have these effects:

- `max_batch_size` caps the requests TensorRT-LLM may schedule together.
- `max_num_tokens` caps the unpadded input tokens in one scheduled batch.
- `max_seq_len` caps prompt plus generated tokens for one request.
- `enable_chunked_prefill: false` disables chunked prompt processing.
- `kv_cache_config.dtype: fp8` uses an FP8 KV cache.
- `enable_block_reuse: false` disables KV block reuse. Keep this disabled when testing Gemma 4 MTP.
- `free_gpu_memory_fraction: 0.5` lets the runtime use half of the memory left after weights and runtime buffers for KV cache.

The model definition remains disabled by default.

## Compiler Concurrency And Caches

TensorRT-LLM, FlashInfer, and TorchInductor may compile kernels when a required variant is not cached. The checked-in model environment bounds that work:

```json
{
  "MAX_JOBS": "4",
  "FLASHINFER_NVCC_THREADS": "1",
  "CMAKE_BUILD_PARALLEL_LEVEL": "4",
  "TORCHINDUCTOR_COMPILE_THREADS": "4"
}
```

The heavy compilation is normally a cold-cache cost for a particular software, GPU, datatype, and kernel combination. Cache hits avoid most compiler work on later starts. Compilation can return after a TensorRT-LLM, CUDA, PyTorch, or FlashInfer upgrade, after cache removal, or when a request reaches a new kernel variant.

Weight loading, KV-cache allocation, warmup, and CUDA graph capture may still run on every start. Those operations are not compiler jobs.

## Validated Gemma 4 Profile

The following path was exercised on 2026-09-01:

| Component | Validated value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, SM120 |
| TensorRT-LLM | `1.3.0rc25` |
| Model | `nvidia/Gemma-4-26B-A4B-NVFP4` from a local checkpoint directory |
| Execution path | TensorRT-LLM PyTorch backend through `trtllm-serve` |
| Result | Model load, health check, text generation, response mapping, and graceful process-group shutdown succeeded |

Image requests were not exercised on the GPU during this validation. The automated tests cover the OpenAI-compatible multimodal payload shape, not TensorRT-LLM image execution.

The leader-already-dead cleanup path is covered by unit tests. A deliberate crash during model build followed by `nvidia-smi` and process-group inspection has not been run manually.

## MTP Speculative Decoding

Upstream TensorRT-LLM documents Gemma 4 MTP through `speculative_config` in the YAML passed to `trtllm-serve`. The llm-pool adapter already passes that YAML, so a future profile can use upstream speculative-decoding fields without adding request-time MTP controls.

The checked-in SM120 profile does not enable MTP. In the local 1.3.0rc25 investigation, an eager SM120 proof of concept ran, but CUDA graphs and batch sizes above one were not ready for this backend. That result was not production-ready, so MTP stayed out of the profile.

This is a validated limitation of the tested combination, not a permanent claim about TensorRT-LLM. Upstream main now documents Gemma 4 target/assistant configurations. Retest the matching Gemma 4 assistant checkpoint after upgrading the runtime. When speculation is enabled upstream, it applies to the server rather than being toggled per llm-pool request.

## Operational Limits

- When no fixed port is configured, llm-pool selects a free port before starting the child. Another process can claim it before TensorRT-LLM binds. Startup errors include the captured output tail so this failure is diagnosable.
- Child stdout and stderr are captured in one anonymous temporary file for the lifetime of the model. The file is not size-bounded or visible by pathname. This is an accepted property until log retention and rotation are designed.
- The child environment forces `PYTHONUNBUFFERED=1`, so a hard crash does not lose the final buffered block of TensorRT-LLM stdout.
- TensorRT-LLM installation, CUDA/OpenMPI installation, and model download are not automated by this repository.
- A slow model load may exceed the deployment helper's default readiness wait. Increase `LLM_POOL_RESTART_WAIT_S` when it is shorter than `trtllm_serve_start_timeout_s`.

## Upstream References

- [NVIDIA TensorRT-LLM Gemma example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/gemma/README.md)
- [NVIDIA TensorRT-LLM speculative decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md)
- [NVIDIA TensorRT-LLM supported models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
- [SM120 Gemma NVFP4 issue for TensorRT-LLM 1.3.0rc22](https://github.com/NVIDIA/TensorRT-LLM/issues/17052)
