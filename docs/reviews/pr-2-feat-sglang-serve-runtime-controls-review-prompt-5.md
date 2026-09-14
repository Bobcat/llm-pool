# Rereview prompt 5: SGLang draft-token UI consistency

Rereview these pull requests together:

- [llm-pool #2](https://github.com/Bobcat/llm-pool/pull/2)
- [llm-workbench #12](https://github.com/Bobcat/llm-workbench/pull/12)

Use the fourth-round findings in
[`pr-2-feat-sglang-serve-runtime-controls-review-findings-4.md`](pr-2-feat-sglang-serve-runtime-controls-review-findings-4.md)
as the review baseline. Review the current branch heads against `main`.

## Required checks

Verify both fourth-round findings:

1. At SGLang top-k greater than 1, editing MTP draft tokens must put
   `sglang_speculative_num_draft_tokens` in the load payload. Top-k 1 must keep
   hiding the independent control.
2. When speculative decoding is disabled in the model definition, the
   definition grid must hide the MTP algorithm, assistant, steps, draft-token,
   and top-k rows.

Confirm that top-k-1 display derivation still shows steps plus 1 and that top-k
greater than 1 still shows the configured width. Recheck the full
cross-repository diff for regressions missed in the previous rounds.

## Validation already performed

- `llm-pool`: `.venv/bin/python -m unittest discover -s tests` — 242 passed.
- `llm-workbench`: `.venv/bin/python -m unittest tests.test_llm_pool_api` — 3
  passed.
- Headless Chromium checks covered the editable top-k-greater-than-1 payload,
  hidden disabled-MTP definition rows, and both draft-token display branches.
- `git diff --check` passed in both repositories.

Write the rereview to:

`docs/reviews/pr-2-feat-sglang-serve-runtime-controls-review-findings-5.md`

Put findings first, ordered by severity. Give every finding an exact file and
line reference. State the disposition of both round-4 findings. End the file
with one of:

- `APPROVE` when no blocking or medium-severity findings remain.
- `REQUEST CHANGES` with the unresolved findings listed first.
