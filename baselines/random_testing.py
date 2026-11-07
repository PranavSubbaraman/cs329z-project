"""Baseline 2: random differential testing (see PREREGISTRATION.md).

Draws specs from the kernel's documented domain (oracle.spaces.sample), runs grad_compare and every property check on
each (torch.compile on every 10th spec only), and stops at the first violation, whose spec becomes the claimed
counterexample, or when the budget is spent. The budget is GPU-seconds of kernel execution (to match an agent run)
and/or a spec count.
Run against one tree:
  KV_KERNEL=tvd KV_TASK_DIR=benchmark/kernels/tvd scripts/harness/in_tree.sh <commit> python baselines/random_testing.py \
      --seed 0 [--gpu-seconds 45] [--max-specs 200] --out results/.../random_tvd@<commit>.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle import PROPERTIES, kernel_name, load_task, replay, sample  # noqa: E402
from scripts.env_report import env_record  # noqa: E402


def run(seed: int, gpu_seconds: float | None, max_specs: int | None, compile_every: int = 10) -> dict:
    k = kernel_name()
    kernel_fn, ref_fn = load_task()
    rng = random.Random(seed)
    used, n, t0 = 0.0, 0, time.time()
    no_compile = [p for p in PROPERTIES if p != "compile_matches_eager"]
    found = None
    log = []
    while True:
        if gpu_seconds is not None and used >= gpu_seconds:
            break
        if max_specs is not None and n >= max_specs:
            break
        spec = sample(k, rng, seed + n)
        checks = ["grad_compare"] + (PROPERTIES if compile_every and n % compile_every == 0 else no_compile)
        t1 = time.time()
        r = replay({"input_spec": spec, "checks": checks}, kernel=k, kernel_fn=kernel_fn, ref_fn=ref_fn)
        dt = time.time() - t1
        used += dt
        n += 1
        log.append({"i": n - 1, "violations": r["violations"], "errors": r["errors"][:2], "seconds": round(dt, 3)})
        if r["reproduces"]:
            found = {"spec": spec, "violations": r["violations"], "spec_index": n - 1,
                     "grad_items": [{x: it.get(x) for x in ("name", "max_rel_err", "ratio", "passed", "reason")}
                                    for it in r.get("grad_compare", {}).get("items", [])],
                     "properties": {p: v.get("status") for p, v in r.get("properties", {}).items()}}
            break
    return {"method": "random_testing", "kernel": k, "tree": os.path.basename(os.environ.get("LIGER_TREE", "")),
            "seed": seed, "budget": {"gpu_seconds": gpu_seconds, "max_specs": max_specs},
            "verdict": "bug" if found else "no_bug", "counterexample": found,
            "specs_run": n, "gpu_seconds_used": round(used, 3), "wall_seconds": round(time.time() - t0, 3),
            "n_spec_errors": sum(bool(e["errors"]) for e in log), "log": log, "env": env_record()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-seconds", type=float, default=None)
    ap.add_argument("--max-specs", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.seed, a.gpu_seconds, a.max_specs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("kernel", "tree", "seed", "verdict", "specs_run", "gpu_seconds_used")}
                     | {"violations": (res["counterexample"] or {}).get("violations")}))


if __name__ == "__main__":
    main()
