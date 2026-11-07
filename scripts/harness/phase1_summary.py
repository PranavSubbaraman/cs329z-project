"""HARNESS-ONLY: collate Phase 1 records into results/phase1_benchmark.{json,md}.

Inputs: harness_only/labels.json (pairs, classes, commits) and harness_only/phase1/<pair_id>.json (repro runs on
buggy/fixed trees and Liger's own test-suite runs, written by the per-kernel builders).
The results/ directory is never copied into an agent sandbox.

    python scripts/harness/phase1_summary.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "harness_only"

# Library-test entries that are diagnostics/transplants/aborted attempts, not the primary count for a commit.
NON_PRIMARY = ("TRANSPLANT", "Diagnostic only", "INCOMPLETE", "Context run", "context run", "diagnostic",
               "interrupted", "Cross-check", "Documented subset")


def primary_libtests(entries, commit):
    for e in entries:
        if not e.get("commit", "").startswith(commit[:8]):
            continue
        blob = (e.get("files", "") + " " + e.get("subset_notes", ""))
        if any(tag in blob for tag in NON_PRIMARY):
            continue
        return e
    return None


def fmt_lib(e):
    if not e:
        return "n/a"
    parts = [f"{e.get('passed', 0)} passed", f"{e.get('failed', 0)} failed"]
    if e.get("skipped"):
        parts.append(f"{e['skipped']} skipped")
    if e.get("errors"):
        parts.append(f"{e['errors']} errors")
    return ", ".join(parts)


def main():
    labels = json.loads((H / "labels.json").read_text())["pairs"]
    rows = []
    for p in labels:
        rec_path = H / "phase1" / f"{p['pair_id']}.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        b, f = rec.get("buggy") or {}, rec.get("fixed") or {}
        lib_b = primary_libtests(rec.get("library_tests", []), p["buggy"])
        lib_f = primary_libtests(rec.get("library_tests", []), p["fixed"]) if p.get("fixed") else None
        buggy_fails = b.get("exit_code") == 1
        fixed_passes = (f.get("exit_code") == 0) if p.get("fixed") else None
        rows.append({
            "pair_id": p["pair_id"], "kernel": p["kernel"], "bug_class": p["bug_class"], "issue": p["issue"],
            "fix_pr": p.get("fix_pr"), "status": p.get("status"), "buggy": p["buggy"][:8],
            "fixed": (p.get("fixed") or "")[:8] or None,
            "backend": (b.get("stdout_json") or {}).get("backend"),
            "repro_buggy_exit": b.get("exit_code"), "repro_fixed_exit": f.get("exit_code") if f else None,
            "reproduces": bool(buggy_fails and (fixed_passes is not False)),
            "reproduces_on_cpu_recorded": rec.get("reproduces_on_cpu"),
            "libtests_buggy": lib_b and {k: lib_b.get(k) for k in ("files", "passed", "failed", "skipped", "errors")},
            "libtests_fixed": lib_f and {k: lib_f.get(k) for k in ("files", "passed", "failed", "skipped", "errors")},
            "libtests_buggy_str": fmt_lib(lib_b), "libtests_fixed_str": fmt_lib(lib_f) if p.get("fixed") else "n/a",
        })
    # GPU validation (scripts/gpu/phase1_gpu.sh on sc: RTX A4000, real Triton kernels), if present.
    gpu_rep = ROOT / "results" / "gpu" / "phase1_repros.jsonl"
    gpu = {}
    if gpu_rep.exists():
        for line in gpu_rep.read_text().splitlines():
            g = json.loads(line)
            gpu[(g["pair"], g["commit"][:8])] = g["exit_code"]
    import re
    def gpu_lib(kernel, commit):
        f = ROOT / "results" / "gpu" / "logs" / f"libtests_{kernel}_{commit[:8]}.log"
        if not f.exists():
            return "n/a"
        m = [l for l in f.read_text().splitlines() if re.search(r" in [0-9.]+s", l) and ("passed" in l or "failed" in l)]
        return re.sub(r",? ?\d+ warnings?", "", m[-1]).split(" in ")[0].strip() if m else "n/a"
    for r, p in zip(rows, labels):
        r["gpu_repro_buggy_exit"] = gpu.get((p["pair_id"], p["buggy"][:8]))
        r["gpu_repro_fixed_exit"] = gpu.get((p["pair_id"], (p.get("fixed") or "")[:8]))
        r["gpu_libtests_buggy"] = gpu_lib(p["kernel"], p["buggy"])
        r["gpu_libtests_fixed"] = gpu_lib(p["kernel"], p["fixed"]) if p.get("fixed") else "n/a"
    out = {"pairs": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "phase1_benchmark.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["| pair | class | issue / fix | buggy -> fixed | repro CPU buggy/fixed | repro GPU buggy/fixed | Liger tests @ buggy (GPU) | Liger tests @ buggy (CPU interp.) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fix = f"PR #{r['fix_pr']}" if r["fix_pr"] else "open"
        pair = f"`{r['buggy']}` -> `{r['fixed']}`" if r["fixed"] else f"`{r['buggy']}` (buggy-only)"
        rep = f"{r['repro_buggy_exit']} / {r['repro_fixed_exit'] if r['fixed'] else 'n/a'}"
        grep = f"{r['gpu_repro_buggy_exit']} / {r['gpu_repro_fixed_exit'] if r['fixed'] else 'n/a'}"
        lines.append(f"| {r['pair_id']} | {r['bug_class']} | #{r['issue']} / {fix} | {pair} | {rep} | {grep} | "
                     f"{r['gpu_libtests_buggy']} | {r['libtests_buggy_str']} |")
    md = "# Phase 1 benchmark (harness view)\n\nRepro exit codes: 1 = bug manifests, 0 = clean.\n\n" + "\n".join(lines) + "\n"
    (ROOT / "results" / "phase1_benchmark.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
