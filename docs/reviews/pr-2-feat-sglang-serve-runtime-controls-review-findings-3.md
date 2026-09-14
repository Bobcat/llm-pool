# Rereview findings 3: SGLang backend and normalized load controls

Targets (branch heads reviewed against `main`, full diff, not only the fix commits):

- `llm-pool` `feat/sglang-serve-runtime-controls` @ `63e9c9c`
  (`63e9c9c fix: pass derived SGLang draft width` on top of `89c1bc9`, `d392f49`)
- `llm-workbench` `feat/llm-pool-sglang-load-controls` @ `c8a25ee`
  (`c8a25ee docs: clarify SGLang draft token ownership` on top of `1f1d456`, `9aad7fc`)

Re-ran `.venv/bin/python -m unittest discover -s tests` in `llm-pool`: 242
passed. Re-ran `.venv/bin/python -m unittest tests.test_llm_pool_api` in
`llm-workbench`: 3 passed. Both fix commits landed while this rereview was in
progress; the findings below are against the committed heads.

All three second-round findings are resolved. One low-severity display
inconsistency remains; it does not block.

---

## 1. Low — The definition grid shows a configured draft-token count the runtime does not use

`llm-workbench` `static/src/workflows/llm-pool/index.js:796`.

```js
      { label: 'MTP draft tokens', value: definition.sglang_speculative_num_draft_tokens, optional: true },
```

The row renders `definition.sglang_speculative_num_draft_tokens` unconditionally.
Now that llm-pool derives the value at top-k 1 rather than rejecting a mismatch,
a definition whose configured pair is stale displays a number the server never
receives. With `sglang_speculative_num_steps: 3` and the default
`sglang_speculative_num_draft_tokens: 6` (`llm-pool app/config.py:146-147`), the
grid shows 6 while the command carries 4.

This is the residue of the round-2 fix: previously such a definition failed to
load, so the stale number was never displayed next to a running model. It is
low severity because the adjacent load-settings note already states the rule
("With top-k 1, llm-pool derives MTP draft tokens as MTP steps + 1",
`index.js:1464`), the editable control is correctly hidden at top-k 1
(`index.js:1418-1427`), and the checked-in profile's 5/6 pair is self-consistent
so nothing ships in this state.

Fix, if you want it: render the derived value when
`definition.sglang_speculative_eagle_topk === 1`, or label the row as configured
rather than effective. Leaving it is defensible — the value is honest about what
`settings.json` says.

---

## Disposition of the second-round findings

| # | Second-round finding | Status |
| --- | --- | --- |
| 1 | Blocking — omitting `--speculative-num-draft-tokens` broke the Frozen-KV MTP profile | **Resolved** |
| 2 | Medium — configured mismatch failed only without a load override | **Resolved** |
| 3 | Low — reserved-flag guards missed `-np` and `--max_num_seqs` | **Resolved** |

**1 — resolved.** `app/engine/sglang_serve.py:271-281` now always emits the
flag, computing the width locally instead of relying on which upstream hook runs:

```python
            speculative_num_draft_tokens = (
                settings.sglang_speculative_num_steps + 1
                if settings.sglang_speculative_eagle_topk == 1
                else settings.sglang_speculative_num_draft_tokens
            )
            command.extend(
                [
                    "--speculative-num-draft-tokens",
                    str(speculative_num_draft_tokens),
                ]
            )
```

This removes the dependency on `_handle_eagle_family` entirely, so the
FROZEN_KV_MTP dispatch (`sglang/srt/speculative/spec_info.py:232-239`) no longer
matters — the value arrives explicitly on every speculative path. Verified
against the checked-in definition:

```
1) checked-in profile (steps=5, topk=1, configured draft=6)
   --speculative-num-draft-tokens = 6

4) topk > 1 keeps the configured draft-token count
   topk=4, draft=8 -> 8
```

The top-k > 1 branch still forwards the configured value, so the fix did not
collapse the two cases. This matches the live validation in the prompt (NEXTN,
five steps, top-k 1, explicit `--speculative-num-draft-tokens 6`, request `OK`).
Covered by `tests/test_engine_sglang_serve.py:81` and the full-command assertion
at `tests/test_engine_sglang_serve.py:260`.

**2 — resolved.** The engine-side rejection is gone; `_command` derives
unconditionally, so the plain-load and override-load paths agree. Verified with
the round-2 reproduction, now symmetric:

```
2) stale configured draft token (steps=3, configured draft=6)
   plain load                   -> 4
   load with unrelated override -> 4
```

An explicitly conflicting API override still fails, at the router
(`app/engine/router.py:1157-1166`):

```
3) explicit conflicting API override (steps=5, draft=4, topk=1)
   ValueError: sglang_speculative_num_draft_tokens must equal
   sglang_speculative_num_steps + 1 when sglang_speculative_eagle_topk is 1
```

The regression test is well chosen: `test_derives_topk_one_draft_token_count`
(`tests/test_engine_sglang_serve.py:81`) feeds a genuinely stale pair
(steps 5, configured draft 4) and asserts `6`, so it would catch a silent
revert to passing the configured value. The override path is covered by
`tests/test_sglang_load_override.py:126`, the rejection by
`tests/test_sglang_load_override.py:144`.

The router still normalises `sglang_speculative_num_draft_tokens` into the
scoped settings (`app/engine/router.py:1167-1170`). That is now redundant with
the engine derivation, but it keeps the scoped `ModelSettings` internally
truthful and has no observable effect — not worth removing.

**3 — resolved.** Both guards were widened:

- `app/engine/llama_server.py:165-170` — `argument in {"-np", "--parallel"}`
  plus the `--parallel=` prefix.
- `app/engine/vllm_serve.py:166-172` — `{"--max-num-seqs", "--max_num_seqs"}`
  plus both `=` prefixes.

Verified by exercising `_command` with each spelling:

```
  llama_server ['--parallel', '8']            -> rejected
  llama_server ['--parallel=8']               -> rejected
  llama_server ['-np', '8']                   -> rejected
  vllm_serve   ['--max-num-seqs', '8']        -> rejected
  vllm_serve   ['--max-num-seqs=8']           -> rejected
  vllm_serve   ['--max_num_seqs', '8']        -> rejected
  vllm_serve   ['--max_num_seqs=8']           -> rejected
  sglang_serve ['--max-running-requests', '8']  -> rejected
  sglang_serve ['--max-running-requests=8']     -> rejected
```

Covered by `tests/test_engine_llama_server.py:63` and
`tests/test_engine_vllm_serve.py:65`, both now table-driven over the spellings.

Two spellings still pass the guards — llama-server `-np=8` and SGLang
`--max_running_requests`. I checked whether either can silently take effect, and
neither can: llama.cpp does not support the `=` form at all (`llama-server
-np=8 -m /nonexistent.gguf` → `error: invalid argument: -np=8`; the long form
`--parallel=8` is rejected the same way), and SGLang parses strictly with
`parser.parse_args(argv)` (`sglang/srt/server_args.py:4273`) while generating no
underscore aliases, so an underscore spelling is an unrecognised argument and
fails startup. Both produce an immediate, legible failure rather than dead
config, so the guards are complete in effect.

---

## Cross-repository check

The round-3 change surface is small — three engine files, one documentation
line, one Workbench string, and the matching tests. I re-checked it in full and
found no new correctness, lifecycle, validation, API, UI, or documentation
regression. Specifically:

- **No behaviour change outside the speculative path.** The SGLang diff is
  confined to `_command`; runtime construction, readiness, shutdown, request
  translation, and error mapping are untouched, and the 11 SGLang lifecycle
  tests still pass.
- **Disabled speculative decoding still emits no flags.** The new `extend` sits
  inside the `if settings.sglang_speculative_algorithm is not None` block
  (`app/engine/sglang_serve.py:264-291`), so
  `test_omits_speculative_flags_when_algorithm_is_disabled` still holds — the
  NEXTN-to-`null` disable path is unaffected.
- **Documentation now matches the implementation.** `docs/runtime-admin-api.md:1001`
  reads "llm-pool derives the draft-token count as the step count plus 1 and
  passes it explicitly", and the Workbench note was corrected in the same
  direction (`index.js:1464`). Both previously credited SGLang with a derivation
  it only performs on the EAGLE path. `README.md` still carries no operational
  memory diagnosis; the 43 GiB note remains in `docs/trtllm-serve-backend.md:105`
  only.
- **Everything resolved in rounds 1 and 2 is still resolved.** Re-verified the
  `target_inflight` constraint/accept/capability triple still naming the same
  five backends (`app/engine/common.py:427-441`,
  `app/engine/router.py:606-613`, `:817-828`), the TensorRT-LLM base-YAML and
  extra-arg `max_batch_size` rejections (`app/engine/trtllm_serve.py:250-259`,
  `:316-319`), and the Workbench payload baseline being `model.definition?.[key]`
  throughout `buildLoadPayload` with no `getEffectiveLoadValue` call left in it.

One unrelated working-tree note, not part of this PR: `llm-workbench` still has
an uncommitted modification to `static/src/workflows/pdf-translation/index.js`.

---

APPROVE

No blocking or medium-severity findings remain. The single low finding — the
definition grid showing a configured draft-token count that a top-k-1 load
derives differently (`llm-workbench static/src/workflows/llm-pool/index.js:796`)
— is optional and can be left as is.
