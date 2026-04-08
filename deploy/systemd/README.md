# systemd

This directory holds systemd-related deploy files for `llm-pool-dev`.

Current scope:
- a user unit
- a start script that supports `DEFAULT_PORT` with optional `service.port` override from `config/settings.json`
- a repo-local `.venv` at `~/projects/llm-pool-dev/.venv`
