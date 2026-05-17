#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/gunnar/projects/llm-pool-dev"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
VENV_BIN="$ROOT_DIR/.venv/bin"
SETTINGS_PATH="${LLM_POOL_SETTINGS_PATH:-${LLM_RESPONSES_API_SETTINGS_PATH:-$ROOT_DIR/config/settings.json}}"
HOST="${HOST:-127.0.0.1}"
DEFAULT_PORT="${DEFAULT_PORT:-8011}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python venv: $PYTHON_BIN" >&2
  exit 127
fi

# Ensure venv console scripts (e.g. ninja) are visible for runtime-loaded extensions.
export PATH="$VENV_BIN:$PATH"

# Keep CUDA extension JIT builds stable.
if [[ -z "${CC:-}" ]] && command -v gcc-12 >/dev/null 2>&1; then
  export CC="gcc-12"
fi
if [[ -z "${CXX:-}" ]] && command -v g++-12 >/dev/null 2>&1; then
  export CXX="g++-12"
fi
if [[ -z "${CUDAHOSTCXX:-}" ]] && command -v g++-12 >/dev/null 2>&1; then
  export CUDAHOSTCXX="g++-12"
fi
if [[ -z "${MAX_JOBS:-}" ]]; then
  export MAX_JOBS="1"
fi

NEEDS_LOCAL_CUDA="$(
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" - "$SETTINGS_PATH" <<'PY'
from __future__ import annotations

import sys

from app.config import load_settings

settings = load_settings(sys.argv[1])
for model_settings in settings.engine.models.values():
    if not model_settings.enabled:
        continue
    backend = model_settings.backend or settings.engine.backend
    if backend.strip().lower() != "openai_compatible":
        print("1")
        break
else:
    print("0")
PY
)"

if [[ "$NEEDS_LOCAL_CUDA" == "1" ]]; then
  # Strict mode: require a Blackwell-capable CUDA toolkit and disallow silent arch fallbacks.
  NVCC_BIN=""
  if [[ -n "${CUDA_HOME:-}" ]] && [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
  elif command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
  fi
  if [[ -z "$NVCC_BIN" ]]; then
    echo "nvcc not found. Refusing to start in strict mode." >&2
    exit 1
  fi
  NVCC_VERSION="$("$NVCC_BIN" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
  if [[ -z "$NVCC_VERSION" ]]; then
    echo "unable to determine nvcc version from: $NVCC_BIN" >&2
    exit 1
  fi
  if ! awk -v v="$NVCC_VERSION" 'BEGIN { split(v, a, "."); exit !((a[1] > 12) || (a[1] == 12 && a[2] >= 8)) }'; then
    echo "nvcc $NVCC_VERSION is too old for strict Blackwell mode (need >= 12.8)." >&2
    exit 1
  fi
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]] && [[ "$TORCH_CUDA_ARCH_LIST" == *"9.0"* ]]; then
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST blocks native Blackwell kernels. Refusing to start." >&2
    exit 1
  fi
fi

PORT="$DEFAULT_PORT"
if [[ -f "$SETTINGS_PATH" ]]; then
  SETTINGS_PORT="$("$PYTHON_BIN" -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); payload=json.loads(p.read_text(encoding='utf-8')); s=payload.get('service',{}) if isinstance(payload,dict) else {}; print(s.get('port','') if isinstance(s,dict) else '')" "$SETTINGS_PATH" 2>/dev/null || true)"
  if [[ -n "$SETTINGS_PORT" ]]; then
    PORT="$SETTINGS_PORT"
  fi
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
