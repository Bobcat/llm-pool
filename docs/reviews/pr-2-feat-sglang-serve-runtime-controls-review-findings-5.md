# Rereview findings 5: SGLang draft-token UI consistency

Targets (branch heads reviewed against `main`):

- `llm-pool` `feat/sglang-serve-runtime-controls` @ `d8694cb` — no code change
  since round 3 (`63e9c9c`); the two commits on top are review documents only.
- `llm-workbench` `feat/llm-pool-sglang-load-controls` @ `47aaa64`
  (`8c94ec8 fix: show effective SGLang draft width`,
  `47aaa64 fix: send SGLang draft width override`).

Re-ran `.venv/bin/python -m unittest discover -s tests` in `llm-pool`: 242
passed. Re-ran `.venv/bin/python -m unittest tests.test_llm_pool_api` in
`llm-workbench`: 3 passed.

## Findings

None. Both fourth-round findings are resolved, the round-3 display derivation
still holds in both branches, and rechecking the full cross-repository diff
surfaced no new correctness, lifecycle, validation, API, UI, or documentation
problem.

---

## Disposition of the round-4 findings

**1 (Medium — draft-token control editable at top-k above 1 but never sent):
resolved.** `static/src/workflows/llm-pool/index.js:2359-2364` puts the key back
in the integer list:

```js
  [
    'sglang_context_length',
    'sglang_max_total_tokens',
    'sglang_speculative_num_steps',
    'sglang_speculative_num_draft_tokens',
  ].forEach((key) => {
```

This is the minimal fix and it lines up with the control gating rather than
duplicating it:

- Top-k above 1: the control renders (`:1437-1442`), the draft is populated, and
  the value is emitted when it differs from `model.definition?.[key]` — the same
  configured-definition baseline every other field uses since round 2. Editing
  8 to 10 now reaches the payload; re-entering 8 correctly stays out of it.
- Top-k 1: the control is still hidden (`:1437-1442` pushes it only when
  `sglangSpeculativeTopk !== 1`), so no draft key is ever created and nothing is
  sent. The requirement that top-k 1 keeps hiding the independent control holds.

The one path that can still produce a top-k-1 draft — a model whose configured
top-k changes from above 1 to 1 through a settings reload while a stale draft
survives `pruneLoadSettingDrafts` (`:213`) — now surfaces as a 400 from
`app/engine/router.py:1157-1166` rather than a silent drop. That is the
behaviour I recommended in round 4 and is strictly better than either
alternative.

`llm-pool` needed no change: it already advertised the constraint
(`app/engine/common.py:670`), accepted the field (`app/schemas.py:267`,
`app/engine/router.py:1084`, `:1097`), and rejects a conflicting explicit value
at top-k 1. The two repositories now agree on the field's contract.

**2 (Low — MTP rows shown with speculative decoding disabled): resolved.**
`static/src/workflows/llm-pool/index.js:781-783` derives the gate and
`:807-813` applies it to all five rows:

```js
      ...(speculativeAlgorithm ? [
        { label: 'MTP algorithm', value: speculativeAlgorithm },
        { label: 'MTP assistant', value: definition.sglang_speculative_draft_model, code: true, optional: true },
        { label: 'MTP steps', value: definition.sglang_speculative_num_steps },
        { label: 'MTP draft tokens', value: speculativeNumDraftTokens },
        { label: 'MTP top-k', value: definition.sglang_speculative_eagle_topk },
      ] : []),
```

`normalizeNullableStringValue` maps `null` and a whitespace-only string to
`null` (`:2084-2088`), so the gate is falsy exactly when the definition has no
algorithm. Confirmed against the payload the pool actually emits:

```
MTP disabled: definition.sglang_speculative_algorithm = None -> grid gate falsy, 5 rows hidden
```

Dropping `optional: true` from the four always-populated rows is right: inside
the guard they can no longer be spurious, and `sglang_speculative_num_steps`,
`sglang_speculative_num_draft_tokens` and `sglang_speculative_eagle_topk` are
plain `int` fields with positive defaults (`app/config.py:146-148`), so they
would never have been hidden by the empty-value test anyway. "MTP assistant"
correctly keeps `optional: true`, since NEXTN without a separate draft
checkpoint is a valid configuration.

---

## Round-3 display derivation, rechecked

Still correct in both branches, and still an exact mirror of the backend rule.
`app/engine/sglang_serve.py:271-275` and
`static/src/workflows/llm-pool/index.js:790-794` apply the same condition to the
same two inputs:

```
backend rule the grid must mirror:
  topk=1 steps=5 cfg=6 -> 6
  topk=1 steps=3 cfg=6 -> 4
  topk=4 cfg=8         -> 8
```

- Top-k 1 shows steps plus 1, including for a stale configured width (3/6 renders
  as 4), matching the headless Chromium result recorded in the prompt.
- Top-k above 1 keeps showing the configured width (8), so the round-4 fix to the
  payload did not collapse the two display branches.

The grid stays a configured-definition view — it derives from
`definition.sglang_speculative_num_steps`, not from an active override — which is
consistent with every other row in it. Unchanged from round 3 and still correct.

---

## Regression check

The round-5 change is confined to two places in one Workbench file: the
definition-grid branch for `sglang_serve` and one entry in the payload list.
Verified it introduces nothing else.

- **No backend change.** `git diff 63e9c9c..HEAD` in `llm-pool` touches only
  `docs/reviews/`. All 242 tests pass.
- **No other payload change.** `buildLoadPayload` (`:2150-2499`) still uses
  `model.definition?.[key]` as the omission baseline throughout, with no
  `getEffectiveLoadValue` call left in it, so the round-2 fix for first-round
  finding 1 (failed-load retry resends unchanged overrides) is intact.
- **Load-settings panel untouched.** The MTP select, the steps control, the
  hidden-at-top-k-1 draft-token control and the derivation note (`:1464`) are all
  unchanged by this round.
- **Earlier rounds still hold.** Re-checked the `target_inflight`
  constraint/accept/capability triple naming the same five backends
  (`app/engine/common.py:427-441`, `app/engine/router.py:606-613`, `:817-828`),
  the TensorRT-LLM extra-arg and base-YAML `max_batch_size` rejections
  (`app/engine/trtllm_serve.py:250-259`, `:316-319`), the widened
  llama-server/vLLM reserved-flag guards (`app/engine/llama_server.py:165-170`,
  `app/engine/vllm_serve.py:166-172`), and the always-emitted derived SGLang
  draft width (`app/engine/sglang_serve.py:264-291`).

Two observations, neither of which I would change:

- **MTP enabled by a one-load override is now invisible in the grid.** For a
  model whose definition has `sglang_speculative_algorithm: null`, the MTP select
  still offers "On", because `configuredSpecAlgorithm` falls back to the
  constraint example (`:1409-1412`, `examples: ["NEXTN"]` at
  `app/engine/common.py:652-657`). Enabling it for one load leaves all five grid rows
  hidden, since the gate reads the definition rather than the effective value.
  That is the correct trade-off: the grid is a definition view, and mixing
  override state into it would break the contract every other row follows. The
  load-settings panel still shows the algorithm, the step count and the
  derivation note for that load, so the effective width remains inferable; only
  top-k is not displayed, and it is pinned at 1 whenever that note appears.
- **No automated guard for the JS behaviour.** `llm-workbench/tests` contains
  only Python API tests; there is no JS harness in the repo, so both round-4
  fixes rest on the headless Chromium checks rather than on a committed
  regression test. That is a pre-existing structural gap, not something this
  round introduced, and out of scope to fix here.

One unrelated working-tree note, not part of this PR: `llm-workbench` still has
an uncommitted modification to `static/src/workflows/pdf-translation/index.js`.

---

APPROVE

Both round-4 findings are resolved, no finding remains at any severity, and all
seven first-round, three second-round, one third-round and two fourth-round
findings are now closed.
