#!/usr/bin/env bash
# Midway demo, simple version (~25 s of runtime). Laptop CPU (Triton interpreter); no GPU or API key needed.
#   PAUSE=1 ./demo_midway.sh      # waits for Enter between steps (for recording)
# Uses recorded agent runs (results/traces/) and replays the agent's counterexample live.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
BUGGY=39adfeef; FIXED=559d33e7
AGENT_BUGGY=results/traces/e97fbfc97da6.json; AGENT_FIXED=results/traces/26e460e4f94d.json
export OMP_NUM_THREADS=4 KV_KERNEL=tvd KV_TASK_DIR=benchmark/kernels/tvd
step() { printf "\n\033[1;36m== %s ==\033[0m\n" "$1"; [ "${PAUSE:-0}" = 1 ] && read -r -p "" _; true; }

step "1. The kernel"
sed -n '89p' work/trees/${BUGGY}*/src/liger_kernel/ops/tvd.py | sed 's/^ */    /'

step "2. Existing checks"
printf "Liger's own test suite:   %s\n" "$(grep -E ' passed' results/gpu/logs/libtests_tvd_${BUGGY}.log | tail -1 | sed 's/, [0-9]* warnings.*//')"
scripts/harness/in_tree.sh $BUGGY $PY baselines/random_testing.py --seed 0 --max-specs 25 --out work/demo/random.json 2>/dev/null \
  | $PY -c "import json,sys; r=json.loads(sys.stdin.read()); print(f\"Random testing:           {r['specs_run']} random inputs -> {'no bug found' if r['verdict']=='no_bug' else 'bug found'}\")"

step "3. The agent"
$PY - "$AGENT_BUGGY" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["record"]; v = r["verdict"]
print(f"Agent: {r['tool_calls']} tool calls, {r['wall_seconds']:.0f} s, ${r['usage']['cost_usd']:.2f}")
print(f"  verdict:  BUG at {v['location']}")
json.dump(v["counterexample"], open("work/demo/agent_counterexample.json", "w"))
EOF
for c in $BUGGY $FIXED; do
  scripts/harness/in_tree.sh $c $PY -c "
import json; from oracle import replay
r = replay({'input_spec': json.load(open('work/demo/agent_counterexample.json')), 'checks': ['grad_compare']})
g = [it for it in r['grad_compare']['items'] if it['name'] == 'grad_p'][0]
side = 'buggy version' if '$c' == '$BUGGY' else 'fixed version'
print(f'  replay on {side}:  ' + (f\"gradient WRONG  (kernel {g['kernel_at']:+.2f}, correct {g['ref_at']:+.2f})\" if r['reproduces'] else 'gradient correct'))
" 2>/dev/null
done

step "4. Fixed version"
$PY - "$AGENT_FIXED" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["record"]; v = r["verdict"]
print(f"Agent: {r['tool_calls']} tool calls, ${r['usage']['cost_usd']:.2f}")
print("  verdict:  NO BUG")
EOF
