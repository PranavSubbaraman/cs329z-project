"""HARNESS-ONLY: print full grad_compare items for a list of specs (JSON file) on one tree."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle import grad_compare, build_inputs, load_task, diff_args, kernel_name
k = kernel_name(); kf, rf = load_task()
for s in json.load(open(sys.argv[1])):
    r = grad_compare(kf, rf, build_inputs(s, diff_args=diff_args(k)))
    for it in r["items"]:
        print(json.dumps({x: it.get(x) for x in ("name", "dtype", "max_abs_err", "ref_max_abs", "err32", "allowed", "ratio", "kernel_at", "ref_at")}))
