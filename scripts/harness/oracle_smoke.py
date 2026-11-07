"""HARNESS-ONLY smoke test of the oracle against one tree: replay a targeted spec and a few random specs.
   KV_KERNEL=tvd KV_TASK_DIR=benchmark/kernels/tvd scripts/harness/in_tree.sh <commit> .venv/bin/python scripts/harness/oracle_smoke.py <spec.json>"""
import json, random, sys, os
sys.path.insert(0, os.getcwd())
from oracle import replay, sample, kernel_name
spec = json.load(open(sys.argv[1]))
r = replay({"input_spec": spec})
gc = r.get("grad_compare", {})
print(json.dumps({"tree": os.path.basename(os.environ.get("LIGER_TREE", "")), "targeted_reproduces": r["reproduces"],
                  "violations": r["violations"], "errors": r["errors"],
                  "grad_items": [{k: it.get(k) for k in ("name", "max_rel_err", "ratio", "passed")} for it in gc.get("items", [])],
                  "props": {p: v["status"] for p, v in r.get("properties", {}).items()}}))
rng = random.Random(1)
k = kernel_name()
res = []
for i in range(int(os.environ.get("N_RANDOM", 6))):
    s = sample(k, rng, i)
    rr = replay({"input_spec": s, "checks": ["grad_compare"]})
    res.append((rr["reproduces"], rr["errors"][:1], [it.get("ratio") for it in rr["grad_compare"].get("items", [])]))
print(json.dumps({"random": res}))
