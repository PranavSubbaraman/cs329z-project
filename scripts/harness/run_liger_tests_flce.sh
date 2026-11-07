#!/usr/bin/env bash
# HARNESS-ONLY: run Liger-Kernel's own FLCE test files at each flce commit on the Triton CPU interpreter.
#   scripts/harness/run_liger_tests_flce.sh [commit-prefix ...]   (default: all four flce commits)
# One pytest invocation per (commit, file), sequentially, -n 3 workers, per-test timeout 600 s.
# Logs: harness_only/phase1/logs/flce_libtests_<prefix>_<file-tag>.log ; a final line "WALL <seconds>" is appended.
# A whole invocation is killed after SUITE_TIMEOUT seconds (default 1800 = the ~30 min timebox).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOGDIR="$ROOT/harness_only/phase1/logs"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-1800}"
EXTRA="${PYTEST_EXTRA:-}"
mkdir -p "$LOGDIR"
commits=("$@")
[ ${#commits[@]} -eq 0 ] && commits=(e44a5fe8 9ac26bb8 395d6166 ae47958e)
for prefix in "${commits[@]}"; do
  tree="$(ls -d "$ROOT"/work/trees/"$prefix"* | head -1)"
  for f in test/transformers/test_fused_linear_cross_entropy.py test/ops/test_fused_linear_cross_entropy.py; do
    [ -f "$tree/$f" ] || continue
    tag="$(echo "$f" | cut -d/ -f2)"
    log="$LOGDIR/flce_libtests_${prefix}_${tag}${LOG_SUFFIX:-}.log"
    cmd=(env TRITON_INTERPRET=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="src:$ROOT/scripts/harness"
         "$PY" -m pytest "$f" -q -p no:cacheprovider -p pytest_cpu_shims -o log_cli=false -n 3 --timeout 600 -rfE $EXTRA)
    echo "# cwd: $tree" > "$log"
    echo "# cmd: ${cmd[*]}" >> "$log"
    start=$(date +%s)
    ( cd "$tree" && "${cmd[@]}" ) >> "$log" 2>&1 &
    pid=$!
    # watchdog for the 30-min timebox
    ( sleep "$SUITE_TIMEOUT"; if kill -0 $pid 2>/dev/null; then echo "TIMEBOX EXCEEDED after ${SUITE_TIMEOUT}s, killing" >> "$log"; pkill -TERM -P $pid; kill -TERM $pid; fi ) &
    wd=$!
    wait $pid; rc=$?
    pkill -P $wd 2>/dev/null; kill $wd 2>/dev/null; wait $wd 2>/dev/null
    end=$(date +%s)
    echo "EXIT $rc" >> "$log"
    echo "WALL $((end-start))" >> "$log"
    echo "$prefix $f rc=$rc wall=$((end-start))s"
  done
done
