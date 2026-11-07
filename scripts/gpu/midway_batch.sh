#!/usr/bin/env bash
# Midway batch on a GPU allocation:
#   oracle  - archive the previous preliminary outputs, re-run calibration (100 specs / fixed commit) and the
#             preliminary random-testing run, both with the current tolerance rule
#   agents  - one full-agent run and one static-review run on every instance (18 runs),
#             3 at a time; per-run JSONL parts are merged into results/runs.jsonl afterwards
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
export TRITON_INTERPRET=0 PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(dirname "$ROOT")/.cache/triton}"
PY="$ROOT/.venv-gpu/bin/python"
case "${1:-}" in
  oracle)
    [ -d results/calibration ] && rm -rf results/calibration_v2_pre_zero_floor && mv results/calibration results/calibration_v2_pre_zero_floor && mv results/calibration_driver.log results/calibration_v2_pre_zero_floor/ 2>/dev/null
    [ -d results/prelim_random ] && rm -rf results/prelim_random_v2_pre_zero_floor && mv results/prelim_random results/prelim_random_v2_pre_zero_floor && mv results/prelim_random_driver.log results/prelim_random_v2_pre_zero_floor/ 2>/dev/null
    scripts/gpu/phase2_gpu.sh 100 > results/calibration_driver.log 2>&1
    OMP_NUM_THREADS=2 scripts/gpu/prelim_random_gpu.sh > results/prelim_random_driver.log 2>&1
    echo "oracle part done" ;;
  agents)
    mkdir -p results/runs_parts
    {
      for i in tvd@39adfeef tvd@559d33e7 flce@395d6166 flce@ae47958e flce@e44a5fe8 flce@9ac26bb8 fljsd@5b828eef fljsd@ed1f856d ce@8c7a5078; do echo "$i full"; done
      for i in tvd@39adfeef tvd@559d33e7 flce@395d6166 flce@ae47958e flce@e44a5fe8 flce@9ac26bb8 fljsd@5b828eef fljsd@ed1f856d ce@8c7a5078; do echo "$i static"; done
    } | xargs -P 3 -L 1 bash -c 'OMP_NUM_THREADS=1 '"$PY"' scripts/run_agent.py --instance "$0" --condition "$1" --tag midway-prelim \
          --runs-file results/runs_parts/"$0"_"$1".jsonl > results/runs_parts/"$0"_"$1".log 2>&1; echo "done $0 $1"'
    cat results/runs_parts/*.jsonl >> results/runs.jsonl
    echo "agents part done" ;;
  *) echo "usage: $0 oracle|agents" ;;
esac
