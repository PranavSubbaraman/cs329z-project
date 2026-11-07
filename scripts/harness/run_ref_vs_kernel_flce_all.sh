#!/usr/bin/env bash
# HARNESS-ONLY: run ref_vs_kernel_flce.py (30 configs, seed 0) on every flce commit, sequentially.
#   must agree:      ae47958e/all, 9ac26bb8/contiguous   (fixed commits; 9ac26bb8 contiguous only, it has the P2 bug)
#   record:          395d6166/all, e44a5fe8/contiguous   (buggy commits)
#   informational:   9ac26bb8/all, e44a5fe8/all, 395d6166/contiguous, ae47958e/contiguous
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$ROOT/harness_only/phase1/logs/flce_ref_vs_kernel.log"
: > "$LOG"
for spec in ae47958e:all 9ac26bb8:contiguous 395d6166:all e44a5fe8:contiguous 9ac26bb8:all e44a5fe8:all 395d6166:contiguous ae47958e:contiguous; do
  c="${spec%%:*}"; lay="${spec##*:}"
  start=$(date +%s)
  out=$("$ROOT/scripts/harness/in_tree.sh" "$c" "$ROOT/.venv/bin/python" "$ROOT/scripts/harness/ref_vs_kernel_flce.py" --layouts "$lay" --n 30 --seed 0 2>/dev/null | grep '^{')
  rc=$?
  echo "$c $lay exit=$rc wall=$(( $(date +%s) - start ))s" >> "$LOG"
  echo "$out" >> "$LOG"
done
echo ALLDONE >> "$LOG"
