#!/usr/bin/env bash
set -euo pipefail

# Restart llm-pool and avoid orphaned vLLM EngineCore processes.
#
# Default behavior:
# - use the user systemd unit when it is installed
# - fall back to the repo-local manual stop/start path otherwise
#
# Usage:
#   restart-llm-pool.sh          # restart on port 8011
#   restart-llm-pool.sh 8013     # use port 8013 for manual mode and health check
#
# Environment:
#   LLM_POOL_RESTART_MODE=auto|systemd|manual  default: auto
#   LLM_POOL_RESTART_WAIT_S=180                readiness wait timeout
#   LLM_POOL_HOST=127.0.0.1                    bind/health host

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT="${1:-8011}"
HOST="${LLM_POOL_HOST:-127.0.0.1}"
MODE="${LLM_POOL_RESTART_MODE:-auto}"
WAIT_S="${LLM_POOL_RESTART_WAIT_S:-180}"
UNIT_NAME="llm-pool.service"
STOP_SCRIPT="$SCRIPT_DIR/stop-llm-pool.sh"
PYTHON_BIN="${LLM_POOL_PYTHON:-$ROOT_DIR/.venv/bin/python}"
ENV_FILE="${LLM_POOL_ENV_FILE:-$HOME/.config/llm-pool/env}"
LOG_DIR="${LLM_POOL_LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LLM_POOL_LOG_FILE:-$LOG_DIR/llm-pool-$PORT.log}"

usage() {
  sed -n '5,18p' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

unit_is_loaded() {
  systemctl --user show "$UNIT_NAME" -p LoadState --value 2>/dev/null | grep -qx "loaded"
}

wait_until_ready() {
  local url="http://$HOST:$PORT/v1/models"
  local last_wait_message=""
  if (( WAIT_S <= 0 )); then
    echo "restart issued; readiness wait disabled"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "restart issued; curl not found, skipping readiness check"
    return 0
  fi

  local deadline=$(( $(date +%s) + WAIT_S ))
  while (( $(date +%s) < deadline )); do
    if [[ "$MODE" == "systemd" || ( "$MODE" == "auto" && "$USING_SYSTEMD" == "1" ) ]]; then
      local active_state=""
      local sub_state=""
      local main_pid=""
      active_state="$(systemctl --user show "$UNIT_NAME" -p ActiveState --value 2>/dev/null || true)"
      sub_state="$(systemctl --user show "$UNIT_NAME" -p SubState --value 2>/dev/null || true)"
      main_pid="$(systemctl --user show "$UNIT_NAME" -p MainPID --value 2>/dev/null || true)"

      if [[ "$active_state" == "failed" ]]; then
        systemctl --user status "$UNIT_NAME" --no-pager --lines=40 || true
        return 1
      fi

      if [[
        -n "${SYSTEMD_OLD_MAIN_PID:-}" &&
        "$SYSTEMD_OLD_MAIN_PID" != "0" &&
        "$main_pid" == "$SYSTEMD_OLD_MAIN_PID"
      ]]; then
        local message="waiting for old llm-pool pid $main_pid to stop ($active_state/$sub_state)"
        if [[ "$message" != "$last_wait_message" ]]; then
          echo "$message"
          last_wait_message="$message"
        fi
        sleep 1
        continue
      fi

      if [[ "$active_state" != "active" || "$sub_state" != "running" || "$main_pid" == "0" ]]; then
        local message="waiting for llm-pool systemd state ($active_state/$sub_state, MainPID=$main_pid)"
        if [[ "$message" != "$last_wait_message" ]]; then
          echo "$message"
          last_wait_message="$message"
        fi
        sleep 1
        continue
      fi
    elif [[ -n "${MANUAL_PID:-}" ]] && ! kill -0 "$MANUAL_PID" 2>/dev/null; then
      echo "llm-pool process exited during startup; recent log:"
      tail -n 80 "$LOG_FILE" 2>/dev/null || true
      return 1
    fi

    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "llm-pool ready: $url"
      return 0
    fi
    local message="waiting for llm-pool health check: $url"
    if [[ "$message" != "$last_wait_message" ]]; then
      echo "$message"
      last_wait_message="$message"
    fi
    sleep 1
  done

  echo "restart issued, but $url was not ready within ${WAIT_S}s"
  if [[ -f "$LOG_FILE" ]]; then
    echo "recent log:"
    tail -n 40 "$LOG_FILE" || true
  fi
  return 1
}

restart_systemd() {
  SYSTEMD_OLD_MAIN_PID="$(systemctl --user show "$UNIT_NAME" -p MainPID --value 2>/dev/null || true)"
  echo "requesting $UNIT_NAME restart via systemd user unit"
  if [[ -n "$SYSTEMD_OLD_MAIN_PID" && "$SYSTEMD_OLD_MAIN_PID" != "0" ]]; then
    echo "old MainPID: $SYSTEMD_OLD_MAIN_PID"
  fi
  systemctl --user restart --no-block "$UNIT_NAME"
  echo "restart job queued"
}

restart_manual() {
  if [[ ! -x "$STOP_SCRIPT" ]]; then
    echo "stop script is not executable: $STOP_SCRIPT" >&2
    exit 1
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "python executable is not available: $PYTHON_BIN" >&2
    exit 1
  fi

  "$STOP_SCRIPT" "$PORT"

  mkdir -p "$LOG_DIR"
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
  export LLM_POOL_SETTINGS_PATH="${LLM_POOL_SETTINGS_PATH:-$ROOT_DIR/config/settings.json}"

  echo "starting llm-pool manually on $HOST:$PORT"
  echo "writing log to $LOG_FILE"
  (
    cd "$ROOT_DIR"
    nohup "$PYTHON_BIN" -m uvicorn app.main:app --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
    echo $!
  ) >"$LOG_DIR/llm-pool-$PORT.pid"
  MANUAL_PID="$(<"$LOG_DIR/llm-pool-$PORT.pid")"
  echo "started llm-pool pid $MANUAL_PID"
}

case "$MODE" in
  auto)
    if unit_is_loaded; then
      USING_SYSTEMD=1
      restart_systemd
    else
      USING_SYSTEMD=0
      restart_manual
    fi
    ;;
  systemd)
    USING_SYSTEMD=1
    restart_systemd
    ;;
  manual)
    USING_SYSTEMD=0
    restart_manual
    ;;
  *)
    echo "invalid LLM_POOL_RESTART_MODE: $MODE" >&2
    echo "expected: auto, systemd, or manual" >&2
    exit 1
    ;;
esac

wait_until_ready
