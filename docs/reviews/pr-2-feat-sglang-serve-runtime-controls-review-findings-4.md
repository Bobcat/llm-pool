# Rereview findings 4: SGLang definition-grid draft tokens

Targets (branch heads reviewed against `main`):

- `llm-pool` `feat/sglang-serve-runtime-controls` @ `63e9c9c` — unchanged since
  round 3; this round touches the Workbench only.
- `llm-workbench` `feat/llm-pool-sglang-load-controls` @ `c8a25ee` plus an
  uncommitted change to `static/src/workflows/llm-pool/index.js` (the round-4
  fix, reviewed from the working tree).

Re-ran `.venv/bin/python -m unittest discover -s tests` in `llm-pool`: 242
passed. Re-ran `.venv/bin/python -m unittest tests.test_llm_pool_api` in
`llm-workbench`: 3 passed.

The round-3 finding is resolved. Rechecking the full cross-repository diff as
the prompt asks surfaced one medium finding that I missed in rounds 2 and 3.

---

## 1. Medium — At top-k above 1 the MTP draft-token control is editable but never reaches the load payload

`llm-workbench` `static/src/workflows/llm-pool/index.js:1429-1437` renders the
control; `:2354-2367` omits the field from the payload.

The control is pushed whenever top-k is not 1:

```js
  const sglangSpecIntegerFields = [
    ['sglang_speculative_num_steps', 'MTP steps'],
  ];
  if (sglangSpeculativeTopk !== 1) {
    sglangSpecIntegerFields.push([
      'sglang_speculative_num_draft_tokens',
      'MTP draft tokens',
    ]);
  }
```

`buildLoadPayload` has no matching emission. The round-2 fix for first-round
finding 2 removed `'sglang_speculative_num_draft_tokens'` from the integer list
and never added it back in any form:

```js
  [
    'sglang_context_length',
    'sglang_max_total_tokens',
    'sglang_speculative_num_steps',
  ].forEach((key) => {
```

Grepping the whole file, `sglang_speculative_num_draft_tokens` now appears only
twice — at `:790` (the new display fallback) and `:1434` (the control key).
There is no payload path at all.

So for a model with `sglang_speculative_eagle_topk` above 1, the user sees an
enabled "MTP draft tokens" number input, edits it, presses Load, and the value
is silently discarded: the model loads with the configured count and nothing
reports the drop. That is the same failure mode as first-round finding 2 — a
control that looks functional and is not — reintroduced in the opposite
direction.

The two repositories disagree here. `llm-pool` advertises the field and accepts
it:

```
load_constraints advertises sglang_speculative_num_draft_tokens: True
topk=4 definition draft tokens: 8
```

(`app/engine/common.py:670` for the constraint, `app/schemas.py:267` for the
request field, `app/engine/router.py:1084`, `:1097` for the accept list.) The
Workbench is the only side that drops it.

Not reachable with any shipped definition — the single checked-in SGLang model
uses top-k 1, where the control is correctly hidden — but top-k above 1 is a
supported, documented configuration, so this is one `settings.json` edit away.

Fix: put the key back in the integer list at `:2354-2357`. The existing control
gating already prevents a top-k-1 draft from being created, and if a stale draft
somehow survives a top-k change, `llm-pool` rejects the conflicting explicit
override with a 400 (`app/engine/router.py:1157-1166`) rather than dropping it
silently — a strictly better outcome than today's behaviour.

I should have caught this in round 2, when the key was removed from the payload
list while the control was only conditionally hidden.

---

## 2. Low — The grid shows MTP rows for a model with speculative decoding disabled

`llm-workbench` `static/src/workflows/llm-pool/index.js:804-808`.

`ModelSettings` gives the speculative fields non-null defaults, so the
definition payload reports them even when the algorithm is off:

```
MTP disabled -> definition payload still reports:
   sglang_speculative_algorithm = None
   sglang_speculative_num_steps = 5
   sglang_speculative_num_draft_tokens = 6
   sglang_speculative_eagle_topk = 1
```

`shouldShowDefinitionField` hides a row only when an `optional` field is empty
(`index.js:828-831`), so "MTP algorithm" disappears while "MTP steps",
"MTP draft tokens" and "MTP top-k" stay visible, describing a path the model
does not take.

Pre-existing since the original Workbench commit, not introduced this round, and
this round's change does not make it worse in kind — it changes the displayed
number from a configured 6 to a derived one. Listing it only because the prompt
asks for regressions missed in earlier rounds. Gating the three rows on
`definition.sglang_speculative_algorithm` would close it; leaving it is
defensible.

---

## Disposition of the round-3 finding

**Resolved.** `static/src/workflows/llm-pool/index.js:779-790` derives the
displayed value and `:807` renders it:

```js
    const speculativeNumDraftTokens = (
      speculativeEagleTopk === 1 && speculativeNumSteps != null
        ? speculativeNumSteps + 1
        : definition.sglang_speculative_num_draft_tokens
    );
```

This matches the backend rule exactly. `app/engine/sglang_serve.py:271-275`
derives `sglang_speculative_num_steps + 1` at top-k 1 and otherwise forwards the
configured count, and the grid now applies the same condition to the same two
inputs. Checked against the definition payloads:

- top-k 1, steps 5 → grid shows 6; `_command` emits
  `--speculative-num-draft-tokens 6`. Agreement.
- top-k 4, configured draft 8 → grid shows 8; `_command` emits 8. The
  "keep displaying the configured count above top-k 1" requirement holds.

The headless Chromium check recorded in the prompt exercises the same two
cases from the rendered page — a stale 3/6 pair showing 4 at top-k 1, and a
top-k-4 definition with width 8 still showing 8 — and agrees with the reading
above.

The two `toPositiveInt` guards are defensive only: `sglang_speculative_num_steps`
and `sglang_speculative_eagle_topk` are non-optional ints in `ModelSettings`
(`app/config.py:146`, `:148` — both plain `int` with positive defaults) and are coerced to positive ints at config load, so
neither is ever null for an SGLang model. For a non-SGLang model the block is
not reached at all.

The grid remains a configured-definition view: it derives from
`definition.sglang_speculative_num_steps`, not from an active load override. That
is consistent with every other row in the grid — "MTP steps" directly above it
shows the configured value too — and the override values are shown by the
load-settings controls instead. No change needed.

---

## Regression check

The round-4 change is display-only and confined to one branch of
`buildLocalDefinitionFields`. Verified it introduces no API, load-payload, or
backend regression:

- **No payload change.** The new `speculativeNumDraftTokens` local is used only
  at `:807`. `buildLoadPayload` (`:2150-2498`) is untouched by this diff and
  still references `model.definition?.[key]` as its omission baseline throughout,
  with no `getEffectiveLoadValue` call left in it — the round-2 fix for
  first-round finding 1 is intact.
- **The editable control is still hidden at top-k 1.** `:1432-1437` is unchanged;
  the derived-display note at `:1464` still fires for that case.
- **No backend change.** `llm-pool` is identical to the round-3 head `63e9c9c`;
  `git log main..HEAD` shows no new commit and the working tree carries only the
  two review documents. All 242 tests pass.
- **Earlier rounds still hold.** Re-checked the `target_inflight`
  constraint/accept/capability triple naming the same five backends
  (`app/engine/common.py:427-441`, `app/engine/router.py:606-613`, `:817-828`),
  the TensorRT-LLM extra-arg and base-YAML `max_batch_size` rejections
  (`app/engine/trtllm_serve.py:250-259`, `:316-319`), the widened
  llama-server/vLLM reserved-flag guards (`app/engine/llama_server.py:165-170`,
  `app/engine/vllm_serve.py:166-172`), and the always-emitted derived SGLang
  draft width (`app/engine/sglang_serve.py:264-291`).

One unrelated working-tree note, not part of this PR: `llm-workbench` still has
an uncommitted modification to `static/src/workflows/pdf-translation/index.js`.

---

REQUEST CHANGES

Unresolved:

1. Medium — at top-k above 1 the MTP draft-token control is editable but never
   reaches the load payload
   (`llm-workbench static/src/workflows/llm-pool/index.js:1429-1437` renders it,
   `:2354-2367` omits it).

The round-3 finding is resolved. Finding 2 is low severity, pre-existing, and
does not block.
