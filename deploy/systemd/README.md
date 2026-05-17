# systemd

This directory holds systemd-related deploy files for `llm-pool-dev`.

Current scope:
- a user unit
- a start script that supports `DEFAULT_PORT` with optional `service.port` override from `config/settings.json`
- default user-unit port `8012`
- a repo-local `.venv` at `~/projects/llm-pool-dev/.venv`
- optional secrets from `~/.config/llm-pool-dev/env`, for example `MOONSHOT_API_KEY=...`
