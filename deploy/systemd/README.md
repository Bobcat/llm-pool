# systemd

This directory holds systemd-related deploy files for `llm-pool`.

Current scope:
- a user unit
- a stop script that terminates the repo-local service process and reaps orphaned vLLM subprocesses
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
