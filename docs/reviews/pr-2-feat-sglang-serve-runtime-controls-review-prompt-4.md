# Rereview prompt 4: SGLang definition-grid draft tokens

Rereview these pull requests together:

- [llm-pool #2](https://github.com/Bobcat/llm-pool/pull/2)
- [llm-workbench #12](https://github.com/Bobcat/llm-workbench/pull/12)

Use the third-round findings in
[`pr-2-feat-sglang-serve-runtime-controls-review-findings-3.md`](pr-2-feat-sglang-serve-runtime-controls-review-findings-3.md)
as the review baseline. Review the current branch heads against `main`.

## Required checks

Verify the remaining low-severity finding from round 3:

1. In the Workbench SGLang definition grid, top-k 1 must display MTP draft
   tokens as steps plus 1. A stale configured draft-token value must not be
   shown as the runtime value. For top-k greater than 1, the grid must keep
   displaying the configured draft-token count.

Confirm that the editable draft-token control remains hidden at top-k 1 and
that this display-only change introduces no API, load-payload, or backend
regression. Recheck the full cross-repository diff for regressions missed in
the previous rounds.

## Validation already performed

- `llm-pool`: `.venv/bin/python -m unittest discover -s tests` — 242 passed.
- `llm-workbench`: `.venv/bin/python -m unittest tests.test_llm_pool_api` — 3
  passed.
- A headless Chromium check used a stale steps/draft pair of 3/6. The top-k-1
  grid showed 4. A top-k-4 definition with draft width 8 still showed 8.
- `git diff --check` passed in both repositories.

Write the rereview to:

`docs/reviews/pr-2-feat-sglang-serve-runtime-controls-review-findings-4.md`

Put findings first, ordered by severity. Give every finding an exact file and
line reference. State whether the round-3 finding is resolved. End the file
with one of:

- `APPROVE` when no blocking or medium-severity findings remain.
- `REQUEST CHANGES` with the unresolved findings listed first.
