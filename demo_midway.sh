#!/usr/bin/env bash
# Midway demo (~2.5 min of runtime). Runs on a laptop CPU (Triton interpreter), no GPU or API key needed.
#   ./demo_midway.sh            # PAUSE=1 ./demo_midway.sh waits for Enter between steps (for recording)
# Needs: .venv (torch + triton), exported trees (python scripts/export_trees.py --kernel tvd; --kernel ce).
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
BUGGY=39adfeef; FIXED=559d33e7
export OMP_NUM_THREADS=4 KV_KERNEL=tvd KV_TASK_DIR=benchmark/kernels/tvd
step() { printf "\n\033[1;36m==== %s ====\033[0m\n" "$1"; [ "${PAUSE:-0}" = 1 ] && read -r -p "(enter) " _; true; }
say()  { printf "\033[0;33m# %s\033[0m\n" "$1"; }
quiet() { grep -v -E "^W1|Redirects are currently not supported|Backend impl" ; }

step "1. The kernel: Liger-Kernel TVD loss, hand-written backward (buggy commit $BUGGY)"
say "The loss is 0.5*|p-q|; the backward writes its gradient with a two-way branch:"
sed -n '85,91p' work/trees/${BUGGY}*/src/liger_kernel/ops/tvd.py

step "2. Liger's own test suite passes on this commit"
say "Recorded GPU run (RTX A4000) of test/transformers/test_tvd.py at $BUGGY:"
grep -E " passed| failed" results/gpu/logs/libtests_tvd_${BUGGY}.log | tail -1

step "3. Random differential testing (fp64 oracle + property checks) also passes"
scripts/harness/in_tree.sh $BUGGY $PY baselines/random_testing.py --seed 0 --max-specs 25 \
  --out work/demo/random_tvd_buggy.json 2>/dev/null
say "Continuous random draws (almost) never produce exact ties p == q."

step "4. Z3: encode the kernel's backward vs. the reference derivative, assert they differ"
$PY scripts/demo/z3_tvd.py buggy
say "SAT with p == q: the kernel gives -s/2 where the derivative of |p-q| is 0."

step "5. Replay the counterexample as tensors (q copies half of p's columns -> ties)"
cat harness_only/targeted_specs/P1_tvd.json | tr -d '\n' | cut -c1-200; echo
for c in $BUGGY $FIXED; do
  printf "tree %s: " "$c"
  scripts/harness/in_tree.sh $c $PY -c "
import json; from oracle import replay
r = replay({'input_spec': json.load(open('harness_only/targeted_specs/P1_tvd.json')), 'checks': ['grad_compare']})
g = [it for it in r['grad_compare']['items'] if it['name'] == 'grad_p'][0]
print('violation' if r['reproduces'] else 'clean', '| grad_p max rel err = %.3g (allowed ratio %.3g)' % (g['max_rel_err'], g['ratio']))
" 2>&1 | quiet
done

step "6. On the fixed commit, Z3 proves equivalence (UNSAT)"
sed -n '86,91p' work/trees/${FIXED}*/src/liger_kernel/ops/tvd.py
$PY scripts/demo/z3_tvd.py fixed

step "7. The same oracle catches other bug classes: a side effect in cross-entropy (open issue)"
say "Input does NOT require grad, some targets are ignore_index -> the kernel should not touch the input."
KV_KERNEL=ce KV_TASK_DIR=benchmark/kernels/ce scripts/harness/in_tree.sh 8c7a5078 $PY -c "
import json; from oracle import replay
r = replay({'input_spec': json.load(open('harness_only/targeted_specs/B5_ce_ignore.json')), 'checks': ['no_mutation_without_grad']})
print('violations:', r['violations']); print(json.dumps(r['properties']['no_mutation_without_grad']['violations'], indent=1))
" 2>&1 | quiet

step "Next: the agent (ReAct over OpenRouter) does steps 4-5 itself from the source, with budgets and baselines"
