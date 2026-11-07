"""Summarize the evaluation: per-instance outcomes and per-method aggregates (the midway report's Table 2).

Reads results/runs.jsonl (agent and static-review runs, already scored by agent/score.py), results/prelim_random/
(random differential testing) and results/gpu/logs/libtests_* (Liger's own test suites), and writes
results/midway_table.{csv,md}.

    python scripts/summarize_results.py [--include-debug]
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["tvd@39adfeef", "tvd@559d33e7", "flce@395d6166", "flce@ae47958e", "flce@e44a5fe8", "flce@9ac26bb8",
         "fljsd@5b828eef", "fljsd@ed1f856d", "ce@8c7a5078"]


def side_of():
    pairs = json.loads((ROOT / "harness_only" / "labels.json").read_text())["pairs"]
    out = {}
    for p in pairs:
        out[p["buggy"][:8]] = (p["pair_id"], "buggy", p["bug_class"])
        if p.get("fixed"):
            out[p["fixed"][:8]] = (p["pair_id"], "fixed", p["bug_class"])
    return out


def liger_tests(inst):
    k, c = inst.split("@")
    f = ROOT / "results" / "gpu" / "logs" / f"libtests_{k}_{c}.log"
    if not f.exists():
        return None
    m = [l for l in f.read_text().splitlines() if re.search(r" in [0-9.]+s", l) and ("passed" in l or "failed" in l)]
    if not m:
        return None
    failed = re.search(r"(\d+) failed", m[-1])
    return "fail" if failed and int(failed.group(1)) > 0 else "pass"


def random_result(inst):
    f = ROOT / "results" / "prelim_random" / f"random_{inst}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"verdict": d["verdict"], "specs_run": d["specs_run"], "gpu_seconds": d["gpu_seconds_used"],
            "violations": (d.get("counterexample") or {}).get("violations")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-debug", action="store_true")
    a = ap.parse_args()
    sides = side_of()
    runs = [json.loads(l) for l in (ROOT / "results" / "runs.jsonl").read_text().splitlines() if l.strip()]
    if not a.include_debug:
        runs = [r for r in runs if not str(r.get("run_tag", "")).startswith("debug")]
    by = defaultdict(list)
    for r in runs:
        by[(r["instance_id"], r["condition"])].append(r)

    rows = []
    for inst in ORDER:
        pair, side, cls = sides[inst.split("@")[1]]
        row = {"instance": inst, "pair": pair, "class": cls, "side": side, "liger_tests": liger_tests(inst)}
        rr = random_result(inst)
        if rr:
            outcome = ("detection" if side == "buggy" else "flagged (replays)") if rr["verdict"] == "bug" else \
                ("miss" if side == "buggy" else "correct_no_bug")
            row["random"] = f"{outcome} ({rr['specs_run']} specs, {rr['gpu_seconds']:.0f} GPU-s)"
        for cond in ("static", "no_z3", "full"):
            rs = by.get((inst, cond), [])
            if rs:
                outs = [r["score"]["outcome"] + ("+loc" if r["score"].get("localized") else "") for r in rs]
                cost = sum(r["usage"]["cost_usd"] for r in rs) / len(rs)
                row[cond] = f"{', '.join(outs)} (${cost:.2f}/run, n={len(rs)})"
        rows.append(row)

    agg = {}
    for cond in ("static", "no_z3", "full"):
        rs = [r for r in runs if r["condition"] == cond]
        if not rs:
            continue
        b = [r for r in rs if r["score"]["side"] == "buggy"]
        f = [r for r in rs if r["score"]["side"] == "fixed"]
        agg[cond] = {"runs": len(rs), "detections": sum(r["score"]["outcome"] == "detection" for r in b), "buggy_runs": len(b),
                     "localized": sum(bool(r["score"].get("localized")) for r in b),
                     "false_alarms": sum(r["score"]["outcome"] == "false_alarm" for r in f), "fixed_runs": len(f),
                     "flagged_real_on_fixed": sum(r["score"]["outcome"] == "possible_new_bug" for r in f),
                     "proofs": sum(r["score"].get("proof") == "proof" for r in rs),
                     "false_proofs": sum(r["score"].get("proof") == "false_proof" for r in rs),
                     "cost_usd": round(sum(r["usage"]["cost_usd"] for r in rs), 3),
                     "mean_gpu_seconds": round(sum(r["gpu_seconds"] for r in rs) / len(rs), 1)}

    cols = ["instance", "class", "side", "liger_tests", "random", "static", "no_z3", "full"]
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "midway_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    md += ["", "| method | runs | detections | localized | false alarms | flagged real bug on fixed | proofs | false proofs | cost $ | mean GPU-s |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for cond, g in agg.items():
        md.append(f"| {cond} | {g['runs']} | {g['detections']}/{g['buggy_runs']} | {g['localized']}/{g['buggy_runs']} | "
                  f"{g['false_alarms']}/{g['fixed_runs']} | {g['flagged_real_on_fixed']} | {g['proofs']} | {g['false_proofs']} | "
                  f"{g['cost_usd']} | {g['mean_gpu_seconds']} |")
    (ROOT / "results" / "midway_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
