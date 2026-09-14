# Review findings: SGLang backend and normalized load controls

Targets:

- `llm-pool` `feat/sglang-serve-runtime-controls` vs `main` (PR #2)
- `llm-workbench` `feat/llm-pool-sglang-load-controls` vs `main` (PR #12)

Re-ran `.venv/bin/python -m unittest discover -s tests` in `llm-pool`: 226 passed.

Findings are ordered by severity. Paths are repo-relative to the repository
named in each finding.

---

## 1. Medium — Retrying a failed load with unchanged values silently drops the overrides

`llm-workbench` `static/src/workflows/llm-pool/index.js:2336`, `:2278`, and the
other `getEffectiveLoadValue` comparisons in `buildLoadPayload`.

The new trtllm/sglang fields only enter the payload when the draft value differs
from `getEffectiveLoadValue(model, key)`, which returns `model.load_override[key]`
when that key is present:

```js
      && value !== toPositiveInt(getEffectiveLoadValue(model, key))
```

`llm-pool` keeps `state.load_override` populated after a *failed* load
(`app/engine/router.py:199` sets it; the failure paths at `app/engine/router.py:215`
and `:238` set `lifecycle = "failed"` without clearing it — only `unload_model`
clears it at `app/engine/router.py:434`). The Workbench draft also survives a load
attempt: `loadSettingDrafts` is only pruned for models that disappear
(`index.js:213`) or cleared when a refresh itself fails (`index.js:222`).

Sequence:

1. User raises `sglang_max_total_tokens` from 20480 to 40960 and loads.
2. The load fails. `load_override = {"sglang_max_total_tokens": 40960}`, state `failed`.
3. The control still shows 40960 (via `getDraftOrEffectiveIntegerValue` →
   `load_override`), and the draft still holds 40960.
4. The user frees VRAM elsewhere and presses Load again without touching the control.
5. `40960 !== toPositiveInt(40960)` is false → the field is omitted → `llm-pool`
   applies `_apply_load_override(..., {})` and loads the **configured** 20480.

The UI shows one value and the runtime uses another, with no error. The same
holds for `trtllm_max_seq_len`, `trtllm_kv_cache_memory_bytes`,
`trtllm_max_num_tokens`, `trtllm_enable_chunked_prefill`, `trtllm_kv_cache_dtype`,
`sglang_context_length`, `sglang_mem_fraction_static`,
`sglang_chunked_prefill_size`, `sglang_kv_cache_dtype`,
`sglang_speculative_num_steps`, and `sglang_speculative_num_draft_tokens`.

The bug class predates this PR (`vllm_max_model_len` and friends use the same
comparison), but this PR adds eleven more fields to it — and the new
`target_inflight` block at `index.js:2142` deliberately compares against
`model.definition?.target_inflight` instead, which is the correct baseline:

```js
    && targetInflight !== toPositiveInt(model.definition?.target_inflight)
```

Fix: compare every load-payload field against `model.definition?.[key]`, matching
what `target_inflight` already does. `definition` is the configured baseline that
an omitted field actually falls back to; `load_override` is the value of a load
that did not succeed.

---

## 2. Medium — SGLang silently overrides the MTP draft-token count when top-k is 1

`llm-pool` `app/engine/router.py:1083-1090` (sglang override validation),
`app/engine/sglang_serve.py:263-271` (flag emission),
`llm-workbench` `static/src/workflows/llm-pool/index.js:1414-1437` (the two
editable MTP integer controls) and `:1452` (the note).

`sglang_speculative_eagle_topk` is accepted as a load override and stays fixed at
1 in the checked-in profile and in the Workbench, which deliberately does not
expose it. With top-k pinned to 1, SGLang derives the draft-token count from the
step count and discards whatever was passed
(`sglang/srt/arg_groups/speculative_hook.py:791-802` in the installed
`sglang-v0519-cu130` build):

```python
    if (
        cfg.speculative_eagle_topk == 1
        and cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        logger.warning(
            "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
        )
```

So "MTP draft tokens" is an editable control that only takes effect when it
happens to equal `steps + 1`. Set steps 5 / draft tokens 4 and the admin API
reports `sglang_speculative_num_draft_tokens: 4` in `load_override`, the UI shows
4, and the server runs 6. The warning lands in the subprocess log, not in any
llm-pool surface.

`app/engine/router.py:1083-1090` validates each speculative field in isolation and
has no cross-field rule.

Fix (pick one):

- reject `sglang_speculative_num_draft_tokens != sglang_speculative_num_steps + 1`
  in the sglang branch of `_apply_load_override` while
  `sglang_speculative_eagle_topk == 1`, so the mismatch is a 400 instead of a
  silent adjustment; or
- stop emitting `--speculative-num-draft-tokens` when top-k is 1 and derive it
  from `--speculative-num-steps`, and drop the Workbench control.

The related `_handle_frozen_kv_mtp` path is fine: it only forces
`max_running_requests=48` when the flag is unset
(`sglang/srt/arg_groups/speculative_hook.py:616-626`), and
`app/engine/sglang_serve.py:282` always passes it, so `target_inflight` wins.

---

## 3. Low — `target_inflight` is advertised as a load control for backends clamped to 1

`llm-pool` `app/engine/common.py:427-434` and `:675`;
`app/engine/router.py:606-613`.

`_load_constraints_for_backend` now merges `common_constraints` into every branch
and returns it as the fallback, so `ct2`, `llama_cpp`, `exllamav3`, `vllm`, and
`stub` all advertise a `target_inflight` constraint. But
`_runtime_capability_for_model` returns 1 for those backends, and the scheduler
uses `min(configured, capability)` (`app/engine/scheduler.py:291`), so the value is
inert.

The Workbench renders the control for all of them
(`static/src/workflows/llm-pool/index.js:884-897`) and only attaches the explanatory
note for the four managed backends (`index.js:901-911`). A `llama_cpp` model
therefore shows an editable "Target inflight", accepts 8, returns HTTP 200, and
runs at 1.

This is documented (`docs/runtime-admin-api.md:293`) and partly mitigated by the
"Effective inflight" row added to the definition grid (`index.js:676`), which is
why it is low rather than medium. Consider either restricting the constraint to
the backends that honour it, or adding a `max` of 1 / a note for the clamped ones
so the UI cannot present a value the runtime will not use.

---

## 4. Low — SGLang lifecycle and error paths are untested

`llm-pool` `tests/test_engine_sglang_serve.py` (2 tests) versus
`tests/test_engine_trtllm_serve.py` (10 tests).

`SglangServeModelRuntime.close()` (`app/engine/sglang_serve.py:50-87`) is a
verbatim copy of the TensorRT-LLM implementation, including the
leader-already-dead branch, the SIGTERM grace loop, the SIGKILL escalation and
the `ProcessLookupError`/`PermissionError` handling. TensorRT-LLM covers all of
that (`tests/test_engine_trtllm_serve.py:300`, `:352`, `:381`, `:420`, `:459`).
SGLang asserts only `killpg(pid, SIGTERM)` once
(`tests/test_engine_sglang_serve.py:257`).

Also missing for SGLang, all of which TensorRT-LLM has:

- `_wait_until_ready` failure → `runtime.close()` (partial-startup cleanup,
  `app/engine/sglang_serve.py:208-212`)
- startup exit / timeout message including the output tail
  (`app/engine/sglang_serve.py:344-362`; cf. `tests/test_engine_trtllm_serve.py:494`)
- empty final content with non-empty `reasoning_content` →
  `sglang_serve_incomplete_response` (`app/engine/sglang_serve.py:551-559`;
  cf. `tests/test_engine_trtllm_serve.py:519`)

Given that the review prompt calls out process-group shutdown and cleanup after
partial startup, the shared logic should be exercised for both backends.

Two smaller test gaps:

- No test that `--max-running-requests` wins over a duplicate in
  `sglang_serve_extra_args`, nor the equivalent for `--parallel` /
  `--max-num-seqs`. See finding 5.
- `test_runtime_config_merges_load_settings_into_base_yaml`
  (`tests/test_engine_trtllm_serve.py:78`) only checks that the base
  `kv_cache_config` keys survive. The docs claim unrelated *top-level* base fields
  are preserved too (`docs/trtllm-serve-backend.md:144`), which the code does
  (`app/engine/trtllm_serve.py:305-310` starts from the loaded mapping) but no test
  pins — add e.g. a `speculative_config:` block to the fixture.

---

## 5. Low — Duplicate native concurrency flags are silently overridden for the three CLI backends, rejected only for TensorRT-LLM

`llm-pool` `app/engine/sglang_serve.py:281-282`,
`app/engine/llama_server.py:201-202`, `app/engine/vllm_serve.py:220-221`, versus
`app/engine/trtllm_serve.py:249-255`.

The normalized flag is appended after `*_extra_args`, so it does win: SGLang and
vLLM both use argparse (last occurrence wins), and llama.cpp's parser re-invokes
the handler per occurrence. That satisfies "the normalized value must win". But
an operator who leaves `--max-num-seqs 4` in `vllm_serve_extra_args` gets no
signal that it is dead config, while the equivalent for TensorRT-LLM raises:

```python
        if any(
            argument in {"--max_batch_size", "--max-batch-size"}
            for argument in settings.trtllm_serve_extra_args
        ):
```

The same asymmetry exists inside the TensorRT-LLM path itself: a base YAML at
`trtllm_serve_config_path` containing `max_batch_size` is silently overwritten by
`app/engine/trtllm_serve.py:287` while the CLI form is rejected.

Suggest either rejecting `--parallel`, `--max-num-seqs`, `--max-running-requests`
in the respective `*_extra_args` (symmetrical with TensorRT-LLM, and cheap), or
documenting explicitly that the extra-args form is ignored.

I verified the TensorRT-LLM YAML path itself is correct against the installed
`trtllm-v130rc25-cu130`: `max_batch_size` is only injected into `llm_args` from
the CLI when it is explicitly passed or non-default
(`tensorrt_llm/commands/serve.py:165-188`), and `kv_cache_config` is deep-merged
with YAML winning (`tensorrt_llm/llmapi/llm_args.py:6399-6408`), so the generated
config's `max_batch_size`, `dtype`, `max_gpu_total_bytes` and the preserved
`enable_block_reuse` / `free_gpu_memory_fraction` all take effect.

---

## 6. Low — The docs state the `target_inflight` → `--max-running-requests` mapping as exact; SGLang may reduce it

`llm-pool` `docs/runtime-admin-api.md:994`, `README.md:449-451`.

SGLang can lower the requested value during KV-cache sizing — see
`sglang/srt/mem_cache/kv_cache_configurator.py:2153`
(`"max_running_requests was reduced from the requested %d to %d"`) and `:2127`.
When that happens llm-pool still admits `target_inflight` concurrently and the
surplus queues inside SGLang. The direction is benign (no over-admission of the
GPU), but the documented invariant "sets both llm-pool admission and the
backend's native concurrency limit" is not guaranteed, and nothing surfaces the
reduction. One sentence in the SGLang load notes would cover it.

---

## 7. Low — Unreachable entries in `isStringLoadSettingKey`

`llm-workbench` `static/src/workflows/llm-pool/index.js:2576-2577`.

Both new entries are dead:

- `sglang_speculative_algorithm` is rendered as a `<select>`
  (`index.js:1404-1412`), and the `change` handler stores
  `loadSettingControl.value` directly (`index.js:325`) without calling
  `parseLoadSettingControlValue`. Only `input[data-load-setting]` reaches that
  function (`index.js:306`).
- `sglang_speculative_draft_model` has no control at all — it is intentionally
  read-only and appears only in the definition grid (`index.js:790`).

Remove both unless a text input for one of them is planned.

---

## Items checked and found correct

- **NEXTN enable/disable round-trip.** `_load_override_payload` iterates
  `model_fields_set` (`app/engine/router.py:774-781`), so an explicit
  `"sglang_speculative_algorithm": null` is preserved rather than dropped as an
  unset default. The Workbench emits it (`index.js:2375-2385`, `''` →
  `normalizeNullableStringValue` → `null`), the proxy forwards the raw dict
  (`llm-workbench app/llm_pool/models.py:124-133`), the router sets the field to
  `None` (`app/engine/router.py:1117-1129`), and `_command` then omits all four
  speculative flags (`app/engine/sglang_serve.py:256-278`), which
  `tests/test_engine_sglang_serve.py:54` covers.
- **`target_inflight` reaching the scheduler.** `_register_executor_group` now
  takes the scoped value (`app/engine/router.py:268-273`) rather than reading
  `_configured_models`, so a one-load override reaches both
  `configured_target_inflight` and `runtime_capability`, and
  `min(configured, capability)` (`app/engine/scheduler.py:291`) equals the native
  server limit for all four managed backends. No over-admission path found.
- **TensorRT-LLM absolute bytes vs fraction.** The installed runtime documents
  exactly the behaviour the docs claim: "If both `max_gpu_total_bytes` and
  `free_gpu_memory_fraction` are specified, memory corresponding to the minimum
  will be allocated" (`tensorrt_llm/llmapi/llm_args.py:3948-3951`). Raising the
  checked-in fraction from 0.5 to 0.9 alongside an 8 GiB absolute budget is
  therefore safe, and `docs/trtllm-serve-backend.md:103` describes it correctly.
- **Generated-YAML cleanup.** Covered on every path I could construct:
  `_command` raising inside the `try` (`app/engine/trtllm_serve.py:199-222`),
  readiness failure via `runtime.close()` (`:233-237`), partial replica load via
  `_cleanup_runtime`, and the double-close guard (`:53-55`). Only a failure inside
  `yaml.safe_dump` itself (`:328-336`, `delete=False`) would leak, which is not
  reachable with the values written there.
- **Checked-in Gemma 4 NVFP4 SGLang definition.** Verified against the installed
  `sglang-v0519-cu130`: `sglang serve` is a real subcommand
  (`sglang/cli/main.py:17`), and `--max-running-requests`, `--max-total-tokens`,
  `--chunked-prefill-size`, `--mem-fraction-static`, `--context-length`,
  `--fp4-gemm-backend`, `--cuda-graph-backend-decode/prefill`, `--tp-size`,
  `--served-model-name` and `modelopt_fp4` all exist. Port is unset (auto-picked),
  `enabled` is `false`, and the compiler-limit env vars mirror the TensorRT-LLM
  profile.
- **README scope.** The 43 GiB allocation diagnosis stays in
  `docs/trtllm-serve-backend.md:104`; `README.md` keeps only the generic
  configuration guidance.
- **Workbench read-only advanced fields.** `sglang_speculative_draft_model`,
  `sglang_speculative_eagle_topk`, `sglang_quantization`,
  `sglang_attention_backend` and `sglang_fp4_gemm_backend` are all present in
  `_BACKEND_MODEL_DEFINITION_FIELDS` (`app/engine/common.py:181-209`) and rendered
  read-only in the definition grid (`index.js:786-797`).
- **Constraint defaults and units.** `getFloatConstraint` falls back to
  `minimum` when `default` is absent (`index.js:1777`), so the missing `default`
  on `sglang_mem_fraction_static` (`app/engine/common.py:621-626`) does not render
  `undefined`. The 256 MiB byte step/minimum for
  `trtllm_kv_cache_memory_bytes` (`app/engine/common.py:591-597`) matches
  `TRTLLM_KV_CACHE_STEP_MIB` in the UI, and the configured 8589934592 bytes is
  on-step.

---

REQUEST CHANGES

Unresolved:

1. Medium — retrying a failed load with unchanged values silently drops the
   overrides (`llm-workbench static/src/workflows/llm-pool/index.js:2336`).
2. Medium — SGLang silently overrides the MTP draft-token count when top-k is 1
   (`llm-pool app/engine/router.py:1083`).

Findings 3-7 are low severity and do not block.
