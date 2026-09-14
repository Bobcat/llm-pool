# Review prompt: SGLang backend and normalized load controls

Review these two pull requests together:

- [llm-pool #2](https://github.com/Bobcat/llm-pool/pull/2): compare
  `feat/sglang-serve-runtime-controls` with `main`.
- [llm-workbench #12](https://github.com/Bobcat/llm-workbench/pull/12): compare
  `feat/llm-pool-sglang-load-controls` with `main`.

The pool change adds a managed `sglang_serve` backend and extends the existing
TensorRT-LLM load controls. It also makes `target_inflight` the normalized
concurrency setting for every managed local server:

- llama-server: `--parallel`
- vLLM Serve: `--max-num-seqs`
- TensorRT-LLM Serve: `max_batch_size`
- SGLang Serve: `--max-running-requests`

The Workbench change exposes the corresponding load-time controls without
editing `settings.json`. It deliberately keeps model paths, binaries,
quantization backends, parser selection, and advanced assistant-model settings
read-only.

## Review priorities

Report correctness bugs, regressions, unsafe lifecycle behavior, and missing
tests. Give each finding a severity and an exact file and line. Check both
repositories before concluding that an API/UI mismatch is harmless.

Focus on:

1. SGLang subprocess startup, readiness, request translation, multimodal input,
   errors, timeouts, process-group shutdown, and cleanup after partial startup.
2. Load override validation and persistence. Confirm that configured values and
   one-load overrides produce the same effective runtime settings returned by
   the admin API.
3. `target_inflight` semantics. Confirm that scheduler admission and native
   server concurrency cannot silently diverge for llama-server, vLLM Serve,
   TensorRT-LLM Serve, or SGLang Serve.
4. TensorRT-LLM YAML merging. Check preservation of unrelated base fields,
   temporary-file cleanup, reserved batch-size handling, and the interaction
   between absolute KV-cache bytes and `free_gpu_memory_fraction`.
5. SGLang MTP. Check `NEXTN` enable/disable behavior, assistant-model handling,
   draft steps, draft-token count, and top-k behavior.
6. Workbench serialization. Check defaults, null values, numeric units and
   steps, enum values, failed-load retry behavior, and whether hidden advanced
   fields remain visible as read-only definition data.
7. Checked-in Gemma 4 NVFP4 definitions. Check backend names, paths, ports,
   memory settings, compiler limits, and disabled/enabled defaults.
8. Documentation claims against the implementation. Keep operational memory
   diagnosis out of the public README.

Pay particular attention to duplicate native CLI flags and precedence. The
normalized value must win, or conflicting configuration must be rejected.

## Validation already performed

- `llm-pool`: `.venv/bin/python -m unittest discover -s tests` — 226 passed.
- `llm-workbench`: `.venv/bin/python -m unittest tests.test_llm_pool_api` — 3
  passed.
- Gemma 4 NVFP4 was loaded through SGLang on an RTX PRO 6000/SM120 and exercised
  with text, image, concurrent, and Frozen-KV MTP requests.
- The final live state was restored to Gemma 4 NVFP4 through vLLM Serve, and a
  non-streaming llm-pool request returned `OK`.

Do not treat unmeasured throughput differences as defects. Report performance
concerns only when they follow from a concrete configuration or code path.

End with one of:

- `APPROVE` when no blocking or medium-severity findings remain.
- `REQUEST CHANGES` with the unresolved findings listed first.
