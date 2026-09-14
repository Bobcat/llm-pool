# Rereview prompt 3: SGLang backend and normalized load controls

Rereview these pull requests together:

- [llm-pool #2](https://github.com/Bobcat/llm-pool/pull/2)
- [llm-workbench #12](https://github.com/Bobcat/llm-workbench/pull/12)

Use the second-round findings in
[`pr-2-feat-sglang-serve-runtime-controls-review-findings-2.md`](pr-2-feat-sglang-serve-runtime-controls-review-findings-2.md)
as the review baseline. Review the current branch heads against `main`; do not
limit the rereview to the latest commits.

## Required checks

Verify that all three second-round findings are resolved:

1. SGLang must always receive `--speculative-num-draft-tokens` when speculative
   decoding is enabled. With top-k 1, llm-pool must derive the value as steps
   plus 1 and pass it explicitly. Confirm this works for the checked-in Gemma 4
   `NEXTN` profile that SGLang promotes to `FROZEN_KV_MTP`.
2. A stale configured draft-token value must not make a plain load behave
   differently from a load with an unrelated override. The command must use the
   derived top-k-1 value in both cases. An explicitly conflicting API override
   must still fail.
3. The reserved native-concurrency guards must reject llama-server `-np` and
   vLLM `--max_num_seqs`, including the underscore flag's `=` form.

Also check the full cross-repository diff for new correctness, lifecycle,
validation, API, UI, or documentation regressions.

## Validation already performed

- `llm-pool`: `.venv/bin/python -m unittest discover -s tests` — 242 passed.
- `llm-workbench`: `.venv/bin/python -m unittest tests.test_llm_pool_api` — 3
  passed.
- The checked-in Gemma 4 SGLang profile started with `NEXTN`, five steps, top-k
  1, and explicit `--speculative-num-draft-tokens 6`. A live request returned
  `OK`.
- SGLang was unloaded after the test. Gemma 4 was restored through vLLM Serve
  with effective `target_inflight` 16, and a live request returned `OK`.
- `git diff --check` passed in both repositories.

Write the rereview to:

`docs/reviews/pr-2-feat-sglang-serve-runtime-controls-review-findings-3.md`

Put findings first, ordered by severity. Give every finding an exact file and
line reference. State the disposition of all three second-round findings. End
the file with one of:

- `APPROVE` when no blocking or medium-severity findings remain.
- `REQUEST CHANGES` with the unresolved findings listed first.
