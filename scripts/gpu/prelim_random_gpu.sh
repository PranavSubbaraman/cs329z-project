#!/usr/bin/env bash
# Preliminary (midway) random-testing run, as preregistered: 200 specs max per instance, seed 0, stop at the first
# violation. Not budget-matched to the agent (that run replaces this one).
#   srun --jobid=<job> --overlap scripts/gpu/prelim_random_gpu.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv-gpu/bin/python"; OUT="$ROOT/results/prelim_random"; mkdir -p "$OUT"
export TRITON_INTERPRET=0 PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(dirname "$ROOT")/.cache/triton}"
for spec in tvd:39adfeef tvd:559d33e7 flce:395d6166 flce:ae47958e flce:e44a5fe8 flce:9ac26bb8 fljsd:5b828eef fljsd:ed1f856d ce:8c7a5078; do
  k="${spec%%:*}"; c="${spec##*:}"
  KV_KERNEL="$k" KV_TASK_DIR="$ROOT/benchmark/kernels/$k" scripts/harness/in_tree.sh "$c" "$PY" baselines/random_testing.py \
    --seed 0 --max-specs 200 --out "$OUT/random_${k}@${c}.json" 2>>"$OUT/stderr.log" | tail -1
done
