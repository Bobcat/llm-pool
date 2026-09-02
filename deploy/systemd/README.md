# systemd

This directory holds systemd-related deploy files for `llm-pool`.

Current scope:
- a user unit
- a stop script that terminates the repo-local service process, captures its managed backend descendants, and separately reaps stray vLLM EngineCore processes
- a restart script that uses the user systemd unit when available, otherwise falls back to manual stop/start, then waits for `/v1/models`
- default user-unit port `8012`
- a repo-local `.venv` at `~/projects/llm-pool/.venv`
- optional secrets from `~/.config/llm-pool/env`, for example `MOONSHOT_API_KEY=...`

Common commands:

```bash
./deploy/systemd/stop-llm-pool.sh
./deploy/systemd/restart-llm-pool.sh
LLM_POOL_RESTART_MODE=manual ./deploy/systemd/restart-llm-pool.sh 8011
```

Managed backend shutdown:

- the systemd unit uses `KillMode=control-group`, so systemd stops llm-pool and its `llama_server`, `trtllm_serve`, and `vllm_serve` child processes together
- manual stop captures descendants before terminating the parent, so managed server processes can still be signalled after reparenting
- the additional stray-process sweep is vLLM-specific; it does not discover a TensorRT-LLM process group after the llm-pool parent is already gone

The restart helper waits up to 180 seconds by default. Increase the readiness wait when an enabled model has a longer startup timeout:

```bash
LLM_POOL_RESTART_WAIT_S=900 ./deploy/systemd/restart-llm-pool.sh
```

Set this value at least as high as the slowest enabled model's startup timeout. A readiness timeout does not itself terminate a model that is still loading.
