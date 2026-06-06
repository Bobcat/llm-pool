# systemd

This directory holds systemd-related deploy files for `llm-pool`.

Current scope:
- a user unit
- a start script that supports `DEFAULT_PORT` with optional `service.port` override from `config/settings.json`
- default user-unit port `8012`
- a repo-local `.venv` at `~/projects/llm-pool/.venv`
- optional secrets from `~/.config/llm-pool/env`, for example `MOONSHOT_API_KEY=...`
