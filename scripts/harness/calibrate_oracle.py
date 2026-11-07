"""HARNESS-ONLY: calibrate / validate the oracle on one tree (run via scripts/harness/in_tree.sh <commit> ...).

  --targeted SPEC.json   replay one harness-only targeted spec (all checks) and print the result
  --n N                  replay N random specs from the kernel's documented domain (oracle.spaces.sample), seeds
                         seed_base .. seed_base+N-1 (calibration uses seed_base 10000, disjoint from evaluation seeds)
Every --compile-every-th random spec also runs the torch.compile property (slow); the others run grad_compare and
the remaining properties. Writes one JSON to --out.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle import PROPERTIES, kernel_name, replay, sample  # noqa: E402
from scripts.env_report import env_record  # noqa: E402


def summarize(r):
    gc = r.get("grad_compare", {})
    return {"reproduces": r["reproduces"], "violations": r["violations"], "errors": r["errors"],
            "items": [{k: it.get(k) for k in ("name", "dtype", "max_rel_err", "ratio", "passed", "reason")}
                      for it in gc.get("items", [])],
            "props": {p: {k: v for k, v in res.items() if k in ("status", "violating_variants", "violations", "nondeterministic", "reason", "backend")}
                      for p, res in r.get("properties", {}).items()},
            "seconds": r["seconds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targeted")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--compile-every", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k = kernel_name()
    out = {"kernel": k, "tree": os.path.basename(os.environ.get("LIGER_TREE", "")), "env": env_record(), "runs": []}
    t0 = time.time()
    if a.targeted:
        spec = json.load(open(a.targeted))
        out["targeted"] = {"spec_file": a.targeted, **summarize(replay({"input_spec": spec}))}
    rng = random.Random(a.seed_base)
    no_compile = [p for p in PROPERTIES if p != "compile_matches_eager"]
    for i in range(a.n):
        s = sample(k, rng, a.seed_base + i)
        checks = ["grad_compare"] + (PROPERTIES if a.compile_every and i % a.compile_every == 0 else no_compile)
        r = replay({"input_spec": s, "checks": checks})
        out["runs"].append({"seed": a.seed_base + i, "spec": s, **summarize(r)})
    ratios = [it["ratio"] for run in out["runs"] for it in run["items"] if isinstance(it.get("ratio"), (int, float))]
    out["summary"] = {"n": a.n, "n_violating": sum(r["reproduces"] for r in out["runs"]),
                      "violations_by_check": {c: sum(c in r["violations"] for r in out["runs"]) for c in ["grad_compare"] + PROPERTIES},
                      "n_errors": sum(bool(r["errors"]) for r in out["runs"]),
                      "max_ratio": max(ratios) if ratios else None,
                      "p99_ratio": sorted(ratios)[int(0.99 * (len(ratios) - 1))] if ratios else None,
                      "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"kernel": k, "tree": out["tree"][:8], "targeted": {x: out.get("targeted", {}).get(x) for x in ("reproduces", "violations", "errors")},
                      "summary": out["summary"]}))


if __name__ == "__main__":
    main()
