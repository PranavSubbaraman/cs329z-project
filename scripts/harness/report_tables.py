"""HARNESS-ONLY: build the preliminary-result tables from results/calibration/*.json and results/prelim_random/*.json.
Writes results/prelim_oracle_and_random.{json,md} (committed) and report/{calib_table,random_table}.tex (local draft).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = {p["pair_id"]: p for p in json.loads((ROOT / "harness_only" / "labels.json").read_text())["pairs"]}
SIDE = {}
for p in LABELS.values():
    SIDE[p["buggy"][:8]] = (p["pair_id"], "buggy")
    if p.get("fixed"):
        SIDE[p["fixed"][:8]] = (p["pair_id"], "fixed")
ORDER = ["39adfeef", "559d33e7", "395d6166", "ae47958e", "e44a5fe8", "9ac26bb8", "5b828eef", "ed1f856d", "8c7a5078"]
KERN = {"39adfeef": "tvd", "559d33e7": "tvd", "395d6166": "flce", "ae47958e": "flce", "e44a5fe8": "flce",
        "9ac26bb8": "flce", "5b828eef": "fljsd", "ed1f856d": "fljsd", "8c7a5078": "ce"}


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    rows_c, rows_r = [], []
    for c in ORDER:
        k = KERN[c]
        pair, side = SIDE[c]
        cal = load(ROOT / "results" / "calibration" / f"{k}_{c}.json")
        if cal:
            t = cal.get("targeted", {})
            s = cal["summary"]
            rows_c.append(dict(instance=f"{k}@{c}", pair=pair, side=side, targeted_violation=t.get("reproduces"),
                               targeted_checks=t.get("violations"), calib_n=s["n"], calib_violating=s["n_violating"],
                               calib_by_check={a: b for a, b in s["violations_by_check"].items() if b},
                               max_ratio=s["max_ratio"], compile_errors=s["n_errors"]))
        rnd = load(ROOT / "results" / "prelim_random" / f"random_{k}@{c}.json")
        if rnd:
            ce = rnd.get("counterexample") or {}
            rows_r.append(dict(instance=f"{k}@{c}", pair=pair, side=side, verdict=rnd["verdict"],
                               violations=ce.get("violations"), spec_index=ce.get("spec_index"),
                               specs_run=rnd["specs_run"], gpu_seconds=rnd["gpu_seconds_used"]))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "prelim_oracle_and_random.json").write_text(json.dumps({"oracle": rows_c, "random": rows_r}, indent=1) + "\n")

    def tf(x):
        return "yes" if x else "no"
    md = ["## Oracle validation (GPU)", "", "| instance | pair/side | targeted spec flagged | violated checks | calibration violations / n | max error/allowed |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[h]\centering\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llccl@{}}\toprule",
           r"Instance & Side & Targeted spec & Random specs violating & Max err/allowed\\\midrule"]
    for r in rows_c:
        cal = f"{r['calib_violating']}/{r['calib_n']}" if r["calib_n"] else "--"
        by = ", ".join(f"{a}:{b}" for a, b in r["calib_by_check"].items())
        md.append(f"| {r['instance']} | {r['pair']} {r['side']} | {tf(r['targeted_violation'])} | {', '.join(r['targeted_checks'] or [])} | {cal} {by} | {r['max_ratio'] if r['max_ratio'] is not None else '--'} |")
        mr = f"{r['max_ratio']:.2f}" if isinstance(r["max_ratio"], float) else "--"
        tex.append(f"{r['instance'].replace('_', '-')} & {r['side']} & {'flagged' if r['targeted_violation'] else 'clean'} "
                   f"({', '.join(x.replace('_', ' ') for x in (r['targeted_checks'] or [])) or '--'}) & {cal}"
                   f"{(' (' + by.replace('_', ' ') + ')') if by else ''} & {mr}" + r"\\")
    tex += [r"\bottomrule\end{tabular}}",
            r"\caption{Oracle validation on GPU. ``Targeted spec'': a harness-side input that triggers the instance's bug "
            r"condition; it must be flagged on the buggy commit and clean on the fixed one. ``Random specs'': 100 draws from "
            r"the documented domain (calibration seeds, disjoint from evaluation). Every 10th also ran torch.compile; "
            r"compile crashes are errors, not violations. Max err/allowed $<1$ means the comparison passed with margin.}",
            r"\end{table}"]
    md += ["", "## Preliminary random differential testing (GPU, 200 specs max, seed 0, stop at first violation)", "",
           "| instance | pair/side | verdict | first violation (spec #) | specs run | GPU-s |", "|---|---|---|---|---|---|"]
    tex2 = [r"\begin{table}[h]\centering\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llllrr@{}}\toprule",
            r"Instance & Side & Verdict & First violation (spec \#) & Specs & GPU-s\\\midrule"]
    for r in rows_r:
        v = f"{', '.join(r['violations'])} (#{r['spec_index']})" if r["violations"] else "--"
        md.append(f"| {r['instance']} | {r['pair']} {r['side']} | {r['verdict']} | {v} | {r['specs_run']} | {r['gpu_seconds']:.1f} |")
        tex2.append(f"{r['instance'].replace('_', '-')} & {r['side']} & {r['verdict'].replace('_', ' ')} & "
                    f"{v.replace('_', ' ').replace('#', chr(92) + '#')} & {r['specs_run']} & {r['gpu_seconds']:.1f}" + r"\\")
    tex2 += [r"\bottomrule\end{tabular}}",
             r"\caption{Preliminary random differential testing (not yet budget-matched to the agent).}", r"\end{table}"]
    (ROOT / "results" / "prelim_oracle_and_random.md").write_text("\n".join(md) + "\n")
    rep = ROOT / "report"
    if rep.exists():
        (rep / "calib_table.tex").write_text("\n".join(tex) + "\n")
        (rep / "random_table.tex").write_text("\n".join(tex2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
