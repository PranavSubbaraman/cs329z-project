#!/usr/bin/env bash
# HARNESS-ONLY: run Liger's own TVD test file at each commit of pair P1_tvd, sequentially, on the Triton CPU
# interpreter. Logs + junit XML go to harness_only/phase1/logs/tvd_liger_tests_<sha8>[_<SUFFIX>].{log,xml}.
# Usage:
#   [NW=2] [KEXPR='<pytest -k expression>'] [SUFFIX=name] [TIMEBOX=seconds] scripts/harness/run_liger_tests_tvd.sh [commit ...]
# Defaults: 2 workers (the heaviest test peaks at ~6.8 GB RSS), no -k filter, no timebox, commits = buggy then fixed.
# History: the full file (no -k) at 39adfeef was interrupted after 25 min (2 timeouts on the 32x4096x1024 shape
# under load avg ~290); the recorded runs use
#   NW=3 KEXPR='41-401-1271 or 3-423-32000 or with_ties' SUFFIX=subset
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOGDIR="$ROOT/harness_only/phase1/logs"
mkdir -p "$LOGDIR"
NW="${NW:-2}"
KEXPR="${KEXPR:-}"
SUFFIX="${SUFFIX:+_$SUFFIX}"
COMMITS=("$@")
[ ${#COMMITS[@]} -gt 0 ] || COMMITS=(39adfeefab06c9a8b596199cb7f9e9d515986d8c 559d33e7b14d81476b295ef41a962937efa74517)
for sha in "${COMMITS[@]}"; do
  tree="$ROOT/work/trees/$sha"
  log="$LOGDIR/tvd_liger_tests_${sha:0:8}${SUFFIX}.log"
  xml="$LOGDIR/tvd_liger_tests_${sha:0:8}${SUFFIX}.xml"
  cmd=(env TRITON_INTERPRET=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="src:$ROOT/scripts/harness"
       "$PY" -m pytest test/transformers/test_tvd.py -q -p no:cacheprovider -p pytest_cpu_shims
       -o log_cli=false -n "$NW" --timeout 600 -rs --durations=25 --junitxml="$xml")
  [ -n "$KEXPR" ] && cmd+=(-k "$KEXPR")
  {
    echo "# commit $sha  tree $tree"
    echo "# started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# cwd $tree"
    printf '# command:'; printf ' %q' "${cmd[@]}"; echo
  } > "$log"
  start=$(date +%s)
  (cd "$tree" && exec "${cmd[@]}") >> "$log" 2>&1 &
  pid=$!
  watcher=""
  if [ -n "${TIMEBOX:-}" ]; then
    # SIGINT makes pytest stop, print its summary and write the junit XML.
    ( sleep "$TIMEBOX"; kill -INT "$pid" 2>/dev/null && echo "# TIMEBOX: sent SIGINT after ${TIMEBOX}s" >> "$log" ) &
    watcher=$!
  fi
  wait "$pid"
  rc=$?
  [ -n "$watcher" ] && kill "$watcher" 2>/dev/null
  end=$(date +%s)
  {
    echo "# exit_code $rc"
    echo "# wall_seconds $((end - start))"
    echo "# finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "$log"
done
