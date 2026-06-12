# Gemma 4 vLLM QAT Notes

This note records the current status of Gemma 4 QAT checkpoints with the
in-process `vllm` backend.

Status: blocked for the official Google 12B QAT checkpoint on the current
runtime.

Checked on 2026-06-09:

- model: `google/gemma-4-12B-it-qat-w4a16-ct`
- local path: `/home/gunnar/models/google/gemma-4-12B-it-qat-w4a16-ct`
- llm-pool id: `gemma-4-12b-it-qat-w4a16-ct-vllm`
- vLLM runtime observed in logs: `v0.22.0`
- quantization path observed in logs: `compressed-tensors`
- kernel path observed in logs: `MarlinLinearKernel for CompressedTensorsWNA16`

The checkpoint downloads and vLLM starts loading it. The checkpoint size is
reported as `9.56 GiB`, and model loading itself takes about `8.0 GiB` of GPU
memory. Engine initialization then fails during the vLLM profiling/warmup run:

```text
RuntimeError: Shape mismatch: a.size(1) = 4096, size_k = 8192
```

The failing call is in Gemma4 unified attention output projection:

```text
attn_output = self.o_proj(attn_output)
```

This is not a KV-cache sizing issue and not an OOM. `vllm_enforce_eager=true`
was tested temporarily and did not change the failure; it still failed in the
same `CompressedTensorsWNA16` / Marlin path.

Community reports
-----------------

The same failure is reported outside this repo:

- `google/gemma-4-12B-it-qat-w4a16-ct` discussion #1 reports the same AOT
  warning and `Shape mismatch: a.size(1) = 4096, size_k = 8192`:
  https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/discussions/1
- `google/gemma-4-12B-it-qat-w4a16-ct` discussion #3 has related vLLM loading
  failures for the Gemma4 unified image and points at native vLLM support /
  fallback-path issues:
  https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/discussions/3
- `Intel/gemma-4-12B-it-int4-AutoRound` discussion #1 reports the same
  `4096` vs `8192` shape mismatch, including with latest vLLM and latest
  Transformers. The Intel maintainer indicates it looks more like a vLLM issue
  than a model issue:
  https://huggingface.co/Intel/gemma-4-12B-it-int4-AutoRound/discussions/1

Current conclusion
------------------

Treat this as a vLLM compatibility issue for Gemma4 unified W4A16 /
compressed-tensors / Marlin, not as an `llm-pool` configuration issue.

The next useful experiment, when acceptable, is to test a newer vLLM build or
Gemma4-specific vLLM container in isolation. Do not carry temporary flags such
as `vllm_enforce_eager` in the model config unless a future runtime proves that
they actually fix this checkpoint.
