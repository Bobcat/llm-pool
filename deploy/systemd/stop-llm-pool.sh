#!/usr/bin/env bash
set -euo pipefail

# Stop a manually launched llm-pool (uv / .venv python) and reap the vLLM
# EngineCore subprocess it spawns.
#
# Manual runs are the problem case: killing the uvicorn parent does NOT reap
# the vLLM EngineCore (spawned via multiprocessing), which keeps holding VRAM.
# The systemd unit does not need this script because its default
# KillMode=control-group already tears down the whole cgroup.
#
# Usage:
#   stop-llm-pool.sh            # stop all llm-pool instances from this repo
#   stop-llm-pool.sh 8013       # stop only the instance on the given port

ROOT_DIR="/home/gunnar/projects/llm-pool"
PORT="${1:-}"
TERM_WAIT_S=8

# Match the uvicorn parent(s) running app.main:app, then keep only those whose
# working directory is this repo. The cwd filter is essential: other services
# (notably llm-workbench) also run `uvicorn app.main:app`, so matching on the
# command line alone would kill them too. cwd also works for relative-path
# launches (e.g. `.venv/bin/python -m uvicorn ...`).
uvicorn_pattern="uvicorn app.main:app"
if [[ -n "$PORT" ]]; then
  uvicorn_pattern="$uvicorn_pattern --host 127.0.0.1 --port $PORT"
fi

parent_pids=()
while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  proc_cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
  if [[ "$proc_cwd" == "$ROOT_DIR" ]]; then
    parent_pids+=("$pid")
  fi
done < <(pgrep -f -- "$uvicorn_pattern" || true)

# Capture vLLM EngineCore descendants BEFORE killing parents. Once the parent
# dies the child is reparented to init (ppid 1) and can no longer be found by
# process tree, so collect them up front.
descendant_pids=()
collect_descendants() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    descendant_pids+=("$child")
    collect_descendants "$child"
  done
}
for pid in "${parent_pids[@]:-}"; do
  [[ -n "$pid" ]] && collect_descendants "$pid"
done

if [[ ${#parent_pids[@]} -eq 0 || -z "${parent_pids[0]:-}" ]]; then
  echo "no manual llm-pool uvicorn process found${PORT:+ on port $PORT}"
else
  echo "stopping llm-pool uvicorn: ${parent_pids[*]}"
  kill -TERM "${parent_pids[@]}" 2>/dev/null || true
fi

# Reap captured descendants plus any stray VLLM::EngineCore. The stray sweep is
# a safety net for orphans left by an earlier unclean stop; on hosts where
# another service also runs vLLM, prefer passing a --port so only the captured
# descendants of this instance are targeted.
mapfile -t stray_engine_pids < <(pgrep -f -- "VLLM::EngineCore" || true)
vllm_pids=("${descendant_pids[@]:-}" "${stray_engine_pids[@]:-}")

# De-duplicate.
declare -A seen=()
unique_vllm_pids=()
for pid in "${vllm_pids[@]:-}"; do
  [[ -z "$pid" ]] && continue
  if [[ -z "${seen[$pid]:-}" ]]; then
    seen[$pid]=1
    unique_vllm_pids+=("$pid")
  fi
done

if [[ ${#unique_vllm_pids[@]} -gt 0 ]]; then
  echo "reaping vLLM subprocesses: ${unique_vllm_pids[*]}"
  kill -TERM "${unique_vllm_pids[@]}" 2>/dev/null || true
fi

# Wait for graceful exit, then SIGKILL whatever is left.
deadline=$(( $(date +%s) + TERM_WAIT_S ))
while (( $(date +%s) < deadline )); do
  still_alive=()
  for pid in "${parent_pids[@]:-}" "${unique_vllm_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && still_alive+=("$pid")
  done
  [[ ${#still_alive[@]} -eq 0 ]] && break
  sleep 0.5
done

for pid in "${parent_pids[@]:-}" "${unique_vllm_pids[@]:-}"; do
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "force killing $pid"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

echo "done"
