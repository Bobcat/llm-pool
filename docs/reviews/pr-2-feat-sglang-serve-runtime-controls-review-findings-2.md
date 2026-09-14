# Rereview findings: SGLang backend and normalized load controls

Targets (branch heads reviewed against `main`, full diff, not only the fix commits):

- `llm-pool` `feat/sglang-serve-runtime-controls` @ `89c1bc9` (fix commit
  `89c1bc9 fix: address managed backend review findings` on top of `d392f49`)
- `llm-workbench` `feat/llm-pool-sglang-load-controls` @ `1f1d456` (fix commit
  `1f1d456 fix: preserve managed backend load overrides` on top of `9aad7fc`)

Re-ran `.venv/bin/python -m unittest discover -s tests` in `llm-pool`: 241
passed. Both fix commits landed while this rereview was in progress; the
findings below are against the committed heads.

Findings are ordered by severity, then the disposition of all seven first-round
findings.

---

## 1. Blocking — Omitting `--speculative-num-draft-tokens` breaks the checked-in Frozen-KV MTP profile

`llm-pool` `app/engine/sglang_serve.py:282-288`.

The fix for first-round finding 2 stops emitting the flag whenever top-k is 1
and relies on SGLang deriving `steps + 1`:

```python
            if settings.sglang_speculative_eagle_topk != 1:
                command.extend(
                    [
                        "--speculative-num-draft-tokens",
                        str(settings.sglang_speculative_num_draft_tokens),
                    ]
                )
```

SGLang only performs that derivation on the **EAGLE** path, not on the
**FROZEN_KV_MTP** path that this PR's flagship profile uses.

In the installed `sglang-v0519-cu130`:

1. `NEXTN` plus a Gemma 4 assistant draft checkpoint is promoted to
   `FROZEN_KV_MTP` — `sglang/srt/arg_groups/speculative_hook.py:69-75`.
2. The per-algorithm dispatch is an `elif` chain —
   `sglang/srt/speculative/spec_info.py:232-239`:
   `is_frozen_kv_mtp()` routes to `_handle_frozen_kv_mtp`, so
   `_handle_eagle_family` never runs.
3. `_handle_frozen_kv_mtp` (`speculative_hook.py:616-637`) touches only
   `max_running_requests` and `enable_mixed_chunk`. It does not set
   `speculative_num_draft_tokens`.
4. The `steps + 1` derivation exists *only* inside `_handle_eagle_family`
   (`speculative_hook.py:791-802`).
5. `speculative_num_draft_tokens` therefore stays at its `None` default
   (`sglang/srt/server_args.py:2106-2110`) and hits
   `assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1`
   (`sglang/srt/speculative/base_spec_worker.py:124`) and
   `assert get_spec().speculative_num_draft_tokens is not None`
   (`sglang/srt/mem_cache/kv_cache_configurator.py:2265`).

Reproduced against the checked-in definition:

```
$ .venv/bin/python -c "<load settings.json, call SglangServeEngine._command>"
algorithm       : NEXTN
draft model     : google/gemma-4-26B-A4B-it-assistant
topk            : 1
num_steps       : 5
num_draft_tokens: 6

speculative flags emitted: ['--speculative-algorithm', '--speculative-num-steps',
                            '--speculative-eagle-topk', '--speculative-draft-model-path']
draft-tokens flag present: False
```

So `gemma-4-26b-a4b-it-nvidia-nvfp4-sglang-serve` — the exact configuration
that was live-validated with Frozen-KV MTP in round 1 — now starts SGLang
without the draft-token width. The round-2 validation list does not include a
live SGLang load, which is why this was not caught: the unit tests assert the
flag's *absence* (`tests/test_engine_sglang_serve.py:81`) rather than the
resulting server behaviour.

This also makes two claims untrue as written:

- `docs/runtime-admin-api.md:1001` — "When top-k is 1, SGLang derives the
  draft-token count as the step count plus 1". True for EAGLE, false for
  FROZEN_KV_MTP.
- `llm-workbench static/src/workflows/llm-pool/index.js:1464` — "With top-k 1,
  SGLang derives MTP draft tokens as MTP steps + 1."

Fix: keep passing the flag, with the derived value, instead of omitting it.
`llm-pool` already knows `steps + 1`; emitting it explicitly satisfies "the
server command must rely on the derived value" and removes the dependency on
which upstream hook happens to run:

```python
            command.extend(
                [
                    "--speculative-num-draft-tokens",
                    str(
                        settings.sglang_speculative_num_steps + 1
                        if settings.sglang_speculative_eagle_topk == 1
                        else settings.sglang_speculative_num_draft_tokens
                    ),
                ]
            )
```

A live load of the checked-in profile should gate this fix, since no unit test
can observe the upstream assertion.

---

## 2. Medium — A configured steps/draft-token mismatch fails only when no load override is present

`llm-pool` `app/engine/sglang_serve.py:232-242` versus
`app/engine/router.py:1149-1171`.

The two validation sites disagree about what to do with the same inconsistency:

- `router._apply_load_override` **normalises**: it rejects only an *explicit*
  `sglang_speculative_num_draft_tokens` in the load body
  (`router.py:1157-1166`), then silently rewrites the value to `steps + 1`
  (`router.py:1167-1170`).
- `sglang_serve._command` **rejects** any mismatch, including one that comes
  purely from `settings.json` (`sglang_serve.py:232-242`).

`_apply_load_override` returns early when the override dict is empty
(`router.py:813-814`), so the normalisation never runs on a plain load. The
result is that the same model definition loads or fails depending on whether an
unrelated override happens to be present:

```
A) load with NO override -> _command sees configured values:
   ValueError: sglang_speculative_num_draft_tokens must equal
   sglang_speculative_num_steps + 1 when sglang_speculative_eagle_topk is 1

B) same config, load WITH an unrelated sglang override (kv_cache_dtype):
   router normalised draft tokens to: 4
   loads fine
```

The trigger is ordinary: `ModelSettings` defaults are
`sglang_speculative_num_steps = 5` and
`sglang_speculative_num_draft_tokens = 6` (`app/config.py:146-147`), so an
operator who sets only `sglang_speculative_num_steps: 3` in `settings.json`
gets a model that refuses to load with no override, and loads with one.

When top-k is 1 the configured draft-token value is unused by definition — it is
derived. Rejecting it is stricter than necessary and creates the split above.
Suggest making the engine derive rather than reject (which finding 1's fix does
anyway), and keeping the hard rejection only for an explicit API override, where
it is already correct.

---

## 3. Low — Reserved-flag rejection misses two real upstream aliases

`llm-pool` `app/engine/llama_server.py:165-170` and
`app/engine/vllm_serve.py:166-171`.

Both guards match only the long dashed spelling and its `=` form:

```python
            argument == "--parallel" or argument.startswith("--parallel=")
            argument == "--max-num-seqs" or argument.startswith("--max-num-seqs=")
```

Two spellings the upstream parsers accept slip through:

- **llama-server `-np`.** Verified against the configured binary:
  `/home/gunnar/.unsloth/llama.cpp/llama-server --help` prints
  `-np,   --parallel N   number of server slots (default: -1, -1 = auto)`.
  `llama_server_extra_args: ["-np", "8"]` is not rejected.
- **vLLM underscore form.** `FlexibleArgumentParser` rewrites `_` to `-` in
  argument names before parsing
  (`vllm/utils/argparse_utils.py:301-315`), so `--max_num_seqs` is a valid vLLM
  flag and is not rejected.

Behaviour is still correct in both cases — the normalized flag is appended last
and wins — so this is a gap in the guard, not a divergence. TensorRT-LLM
(`app/engine/trtllm_serve.py:250-259`) and SGLang
(`app/engine/sglang_serve.py:224-231`) have no equivalent alias to miss;
SGLang's parser generates no underscore aliases.

Adding `-np` and the underscore spellings to the two match sets closes it.

---

## Disposition of the first-round findings

| # | First-round finding | Status |
| --- | --- | --- |
| 1 | Failed-load retry silently drops overrides | **Resolved** |
| 2 | MTP draft tokens silently overridden at top-k 1 | **Resolved, with a blocking regression** (new finding 1) and a new inconsistency (new finding 2) |
| 3 | `target_inflight` advertised for clamped backends | **Resolved** |
| 4 | SGLang lifecycle/error tests missing | **Resolved** |
| 5 | Duplicate native concurrency flags not rejected | **Resolved**, with an alias gap (new finding 3) |
| 6 | Docs state the SGLang mapping as exact | **Resolved** |
| 7 | Dead `isStringLoadSettingKey` entries | **Resolved** |

**1 — resolved.** Every comparison in `buildLoadPayload` now uses
`model.definition?.[key]` as the omission baseline; no `getEffectiveLoadValue`
call remains in the function (`index.js:2139-2487`). The fix was applied beyond
the new fields to the pre-existing vLLM and llama-server ones
(`index.js:2199`, `:2214`, `:2224`, `:2235`, `:2247`, `:2259`, `:2273`, `:2286`,
`:2404-2444`), matching the `gguf_*`/`exllama_*` blocks that already did this.
`getEffectiveLoadValue` correctly survives on the display path
(`index.js:1829-1877`), so a failed load still shows what was attempted while
the payload resends it. `getMmProcessorMaxPixels` was split so the
`vllm_max_pixels` comparison uses the configured value
(`index.js:1963-1970`, `:2224`).

**2 — resolved as specified, but see findings 1 and 2.** The Workbench no
longer offers an independent draft-token control at top-k 1
(`index.js:1415-1427`), a conflicting explicit API value is rejected
(`router.py:1157-1166`), and the derived value is used. The mechanism chosen for
"use the derived value" is the regression in finding 1.

**3 — resolved.** `_load_constraints_for_backend` now emits the
`target_inflight` constraint only for `llama_server`, `openai_remote`,
`sglang_serve`, `trtllm_serve` and `vllm_serve`
(`app/engine/common.py:427-441`), matching `_runtime_capability_for_model`
(`app/engine/router.py:606-613`) exactly. `_apply_load_override` rejects the
field for every other backend before popping it (`app/engine/router.py:817-828`),
so a client that sends it anyway gets a 400 rather than a silent no-op. Covered
by `test_target_inflight_load_override_is_rejected_for_clamped_backend` and
`test_vllm_serve_also_exposes_target_inflight`. Docs updated consistently
(`docs/runtime-admin-api.md:19`, `:390`, `:830-837`). Since the Workbench gates
every control on `hasLoadConstraint`, the control now disappears for clamped
backends with no UI change needed, while the read-only "Target inflight" /
"Effective inflight" rows stay visible for all backends
(`index.js:666`, `:676`).

**4 — resolved.** `tests/test_engine_sglang_serve.py` grew from 2 to 11 tests
and now covers the paths the prompt names: readiness-failure cleanup
(`:308`), process-group SIGTERM/SIGKILL escalation (`:342`), leader-already-dead
escalation (`:361`), orphan group gone or not permitted (`:378`), survival past
SIGKILL (`:407`), startup exit with the output tail (`:430`), and incomplete
responses (`:448`). That is parity with the TensorRT-LLM set. The reserved-flag
paths gained direct tests in all four backends
(`tests/test_engine_llama_server.py`, `test_engine_vllm_serve.py`,
`test_engine_sglang_serve.py:57`, `test_engine_trtllm_serve.py`), including
`test_runtime_config_rejects_base_max_batch_size`.

**5 — resolved, with the alias gap in finding 3 above.** All four backends now
reject the reserved flag rather than silently overriding it
(`llama_server.py:165-170`, `vllm_serve.py:166-171`,
`sglang_serve.py:224-231`, `trtllm_serve.py:250-259`), the `--flag=value` form
is matched everywhere, and TensorRT-LLM additionally rejects `max_batch_size` in
the base YAML (`trtllm_serve.py:316-319`) — closing the asymmetry where the CLI
form raised and the YAML form was silently overwritten. The rejection fires
before the temporary YAML is created, so nothing leaks. The checked-in
`config/settings.json` and `config/trtllm/gemma-4-26b-a4b-nvfp4.yaml` carry none
of the reserved flags, so no shipped profile regresses.

**6 — resolved.** The mapping is now described as a request rather than a
guarantee, with the reduction called out:
"`target_inflight` is passed to SGLang as `--max-running-requests` ... SGLang may
reduce its native limit during KV-cache sizing; excess admitted requests then
wait inside SGLang" (`docs/runtime-admin-api.md:997`). The same wording change
was applied consistently across `README.md:449-452`,
`docs/model-replica-routing-notes.md:13`, `:282`,
`docs/runtime-scheduler-notes.md:20`, `docs/runtime-scheduler-tracker.md:17`,
and `docs/runtime-subprocess-notes.md:15`. `README.md` still carries no
operational memory diagnosis — the 43 GiB allocation note remains in
`docs/trtllm-serve-backend.md:105` only. New "do not set this flag" lines were
added at `README.md:822`, `:950`, `docs/runtime-admin-api.md:981`, `:1002`,
`:1016`, and `docs/trtllm-serve-backend.md:62`.

**7 — resolved.** Both SGLang entries are gone from `isStringLoadSettingKey`
(`index.js:2585-2591`).

---

## Cross-repository consistency

No API/UI mismatch found in the fixed state. Spot-checked:

- `target_inflight` constraint presence (`common.py:427-441`) versus the
  Workbench control gate (`index.js:884-897`) versus the router's accept list
  (`router.py:817-828`) — all three name the same five backends.
- The Workbench reads `sglang_speculative_eagle_topk` from the definition
  payload (`index.js:1415-1417`); the field is in
  `_BACKEND_MODEL_DEFINITION_FIELDS` (`common.py:197`), and `ModelSettings`
  defaults it to 1, so the top-k-1 branch is taken for every SGLang model. For a
  non-SGLang model the lookup yields `null`, the `!== 1` branch is taken, and
  `getIntegerConstraint` then returns `null`, so nothing is rendered — safe.
- `AdminLoadRequest` still carries `sglang_speculative_num_draft_tokens`
  (`app/schemas.py:267`), which is correct: it stays reachable for a top-k > 1
  configuration and is rejected on conflict at top-k 1.

One unrelated working-tree note, not part of this PR: `llm-workbench` has an
uncommitted modification to `static/src/workflows/pdf-translation/index.js`.

---

REQUEST CHANGES

Unresolved:

1. Blocking — omitting `--speculative-num-draft-tokens` at top-k 1 leaves
   `speculative_num_draft_tokens` unset on SGLang's FROZEN_KV_MTP path, breaking
   the checked-in Gemma 4 NVFP4 profile
   (`llm-pool app/engine/sglang_serve.py:282-288`).
2. Medium — a configured steps/draft-token mismatch fails only when no load
   override is present (`llm-pool app/engine/sglang_serve.py:232-242` versus
   `app/engine/router.py:1149-1171`).

Finding 3 is low severity and does not block. All seven first-round findings are
addressed; findings 1 and 2 above are regressions introduced by the fix for
first-round finding 2, and finding 1 needs a live load of the checked-in SGLang
profile to confirm the correction.
