#!/usr/bin/env bash
# Create the Linux/CUDA venv (.venv-gpu) from requirements.txt. On Linux, torch 2.15.0 pulls its matching triton wheel,
# so no source build is needed. Caches go under the repo's parent dir (e.g. /atlas/u/$USER), not the home quota.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$ROOT")/.cache/pip}"
export TMPDIR="${TMPDIR:-$(dirname "$ROOT")/.cache/tmp}"; mkdir -p "$TMPDIR"
[ -x .venv-gpu/bin/python ] || python3 -m venv .venv-gpu
.venv-gpu/bin/pip install -q --upgrade pip
.venv-gpu/bin/pip install -q -r requirements.txt
.venv-gpu/bin/pip freeze | grep -iE '^(torch|triton|nvidia-cuda-runtime|numpy|z3-solver|openai)' || true
TRITON_INTERPRET=0 .venv-gpu/bin/python scripts/env_report.py results/gpu/env_gpu.json
