#!/usr/bin/env bash
# Phase 2 oracle validation on a CUDA GPU: targeted harness specs on every instance (the oracle must flag each bug
# on its buggy tree and nothing on the fixed tree), plus random-spec calibration on the fixed trees.
#   srun --jobid=<job> --overlap scripts/gpu/phase2_gpu.sh [N_CALIB]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv-gpu/bin/python"; OUT="$ROOT/results/calibration"; mkdir -p "$OUT"
export TRITON_INTERPRET=0 PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(dirname "$ROOT")/.cache/triton}"
N="${1:-100}"
TS="$ROOT/harness_only/targeted_specs"
run() {  # kernel commit targeted_spec n
  KV_KERNEL="$1" KV_TASK_DIR="$ROOT/benchmark/kernels/$1" scripts/harness/in_tree.sh "$2" "$PY" scripts/harness/calibrate_oracle.py \
    ${3:+--targeted "$3"} --n "$4" --out "$OUT/${1}_${2}.json" 2>>"$OUT/stderr.log" | tail -1
}
run tvd 39adfeef "$TS/P1_tvd.json" 0
run tvd 559d33e7 "$TS/P1_tvd.json" "$N"
run flce 395d6166 "$TS/P2_flce_strides.json" 0
run flce ae47958e "$TS/P2_flce_strides.json" "$N"
run flce e44a5fe8 "$TS/P3_flce_none.json" 0
run flce 9ac26bb8 "$TS/P3_flce_none.json" "$N"
run fljsd 5b828eef "$TS/P4_fljsd_bf16.json" 0
run fljsd ed1f856d "$TS/P4_fljsd_bf16.json" "$N"
run ce 8c7a5078 "$TS/B5_ce_ignore.json" "$N"
