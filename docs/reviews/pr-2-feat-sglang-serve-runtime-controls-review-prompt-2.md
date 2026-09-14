# Rereview prompt: SGLang backend and normalized load controls

Rereview these pull requests together:

- [llm-pool #2](https://github.com/Bobcat/llm-pool/pull/2)
- [llm-workbench #12](https://github.com/Bobcat/llm-workbench/pull/12)

Use the first-round findings in
[`pr-2-feat-sglang-serve-runtime-controls-review-findings-1.md`](pr-2-feat-sglang-serve-runtime-controls-review-findings-1.md)
as the review baseline. Check the current branch heads against `main`; do not
limit the rereview to the latest commits.

## Required checks

Verify that every first-round finding is resolved without a regression:

1. A failed-load retry must resend unchanged runtime overrides. Workbench
   payload comparisons must use the configured definition as the omission
   baseline, not the failed load override currently displayed.
2. With SGLang speculative top-k 1, draft tokens must equal steps plus 1. The
   server command must rely on the derived value, conflicting explicit API
   values must fail, and Workbench must not expose an independent draft-token
   control for that case.
3. `target_inflight` must only be advertised and accepted as a load override by
   backends that can use concurrent admission. Confirm the managed server
   mappings and `openai_remote`; confirm clamped in-process backends do not show
   the control.
4. SGLang startup failure, output-tail reporting, incomplete responses, and
   process-group shutdown must have direct tests.
5. Native concurrency settings in extra arguments must be rejected for
   llama-server, vLLM Serve, and SGLang. TensorRT-LLM must reject both extra-arg
   and base-YAML `max_batch_size` conflicts, including `--flag=value` forms.
6. Documentation must describe SGLang concurrency as a requested native limit
   because SGLang may lower it during KV-cache sizing. Keep operational memory
   diagnosis out of the public README.
7. The dead SGLang entries in `isStringLoadSettingKey` must be gone.

Also check cross-repository API/UI consistency and any new correctness,
lifecycle, or validation problem introduced by the fixes.

## Validation already performed

- `llm-pool`: `.venv/bin/python -m unittest discover -s tests` — 241 passed.
- `llm-workbench`: `.venv/bin/python -m unittest tests.test_llm_pool_api` — 3
  passed.
- A headless Chromium test performed a failing SGLang load followed by Retry.
  Both requests carried the same `sglang_max_total_tokens` override.
- `git diff --check` passed in both repositories.

Write the rereview to:

`docs/reviews/pr-2-feat-sglang-serve-runtime-controls-review-findings-2.md`

Put findings first, ordered by severity. Give every finding an exact file and
line reference. State the disposition of all seven first-round findings. End
the file with one of:

- `APPROVE` when no blocking or medium-severity findings remain.
- `REQUEST CHANGES` with the unresolved findings listed first.
