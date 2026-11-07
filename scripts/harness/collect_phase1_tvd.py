"""HARNESS-ONLY: assemble harness_only/phase1/P1_tvd.json.

Runs the repro harness_only/repros/P1_tvd.py on the buggy and the fixed tree (via scripts/harness/in_tree.sh),
recording its stdout JSON line, exit code and wall seconds, and parses the Liger test-suite logs written by
scripts/harness/run_liger_tests_tvd.sh (plus the cross-check log) into library_tests entries, with the counts
exactly as pytest's final summary line reports them.

Run:  .venv/bin/python scripts/harness/collect_phase1_tvd.py
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
LOGS = ROOT / "harness_only" / "phase1" / "logs"
BUGGY = "39adfeefab06c9a8b596199cb7f9e9d515986d8c"
FIXED = "559d33e7b14d81476b295ef41a962937efa74517"
REPRO = ROOT / "harness_only" / "repros" / "P1_tvd.py"
OUT = ROOT / "harness_only" / "phase1" / "P1_tvd.json"

SUBSET_K = "41-401-1271 or 3-423-32000 or with_ties"
SUBSET_NOTE = (
    "Documented subset (-k '" + SUBSET_K + "'): the full file did not fit the ~30 min timebox on the CPU "
    "interpreter (see the interrupted full run). Kept: all three pre-existing test functions x all 4 reductions x "
    "both dtype params x every ignore_index, for the two pre-existing shapes with the fewest kernel programs, "
    "(41,401,1271) and (3,423,32000), plus (fixed commit only) all 32 new test_correctness_with_ties cases. "
    "Dropped: shapes (1,4096,32000) and (32,4096,1024) (4096 and 131072 rows = interpreter programs; the latter "
    "timed out at 600 s per test under load) and (1,4096,128256) (skipped anyway: shim reports 32 GiB < 36e9). "
    "On CPU every bfloat16 case is skipped by Liger's own supports_bfloat16() (False for device 'cpu'), so only "
    "float32 cases execute. Selection (pytest --collect-only with the same -k): 80 of 200 tests at 39adfeef, "
    "112 of 232 at 559d33e7; under xdist pytest's summary line omits the deselected count."
)
LIBRARY_LOGS = [
    {
        "log": f"tvd_liger_tests_{BUGGY[:8]}_full_interrupted.log",
        "commit": BUGGY,
        "files": "test/transformers/test_tvd.py",
        "subset_notes": (
            "FULL file (200 collected), -n 2, interrupted by SIGINT after 24:55 to respect the ~30 min timebox "
            "(machine load average ~290 from concurrent builders). Counts are pytest's summary at interruption. "
            "Both failures are pytest-timeout (>600 s) on shape (32,4096,1024) (131072 interpreter programs), "
            "test_correctness[dtype1-1e-08-1e-06-mean-32-4096-1024] and "
            "test_correctness_not_last[dtype1-1e-08-1e-06-batchmean-32-4096-1024], i.e. not assertion failures."
        ),
    },
    {
        "log": f"tvd_liger_tests_{BUGGY[:8]}_subset.log",
        "commit": BUGGY,
        "files": "test/transformers/test_tvd.py",
        "subset_notes": SUBSET_NOTE,
    },
    {
        "log": f"tvd_liger_tests_{FIXED[:8]}_subset.log",
        "commit": FIXED,
        "files": "test/transformers/test_tvd.py",
        "subset_notes": SUBSET_NOTE,
    },
    {
        "log": f"tvd_liger_tests_{FIXED[:8]}_full.log",
        "commit": FIXED,
        "files": "test/transformers/test_tvd.py",
        "subset_notes": "FULL file (232 collected), retried at lower machine load, -n 2, timebox 1800 s.",
    },
    {
        "log": f"tvd_liger_tests_{BUGGY[:8]}_full.log",
        "commit": BUGGY,
        "files": "test/transformers/test_tvd.py",
        "subset_notes": "FULL file (200 collected), re-run by the lead on an idle machine after the builder stopped "
                        "(usage limit): -n 4, OMP_NUM_THREADS=1, --timeout 600 per test, no overall timebox.",
    },
    {
        "log": f"tvd_liger_tests_xcheck_{FIXED[:8]}tests_on_{BUGGY[:8]}.log",
        "commit": BUGGY,
        "files": f"test/transformers/test_tvd.py from {FIXED[:8]} (copied to scratch), -k with_ties",
        "subset_notes": (
            f"Cross-check: the fixed commit's new regression tests (test_correctness_with_ties, 32 cases) run "
            f"against the BUGGY source: cwd, rootdir and pytest config = the {BUGGY[:8]} tree, so liger_kernel and "
            f"test.utils resolve to {BUGGY[:8]} (verified by a printing plugin: the kernel line is the two-way "
            f"tl.where). bfloat16 cases skip on CPU. PR #1374 reports 24 of 32 failing on H100 when only the "
            f"kernel change is reverted (8 reduction='none' cases pass because they skip backward)."
        ),
    },
]

COUNT_KEYS = ["passed", "failed", "skipped", "errors", "xfailed", "xpassed", "deselected", "warnings"]


def parse_log(path):
    text = path.read_text()
    cmd = next((l[len("# command:"):].strip() for l in text.splitlines() if l.startswith("# command:")), None)
    cwd = next((l[len("# cwd"):].strip() for l in text.splitlines() if l.startswith("# cwd")), None)
    wall = next((int(l.split()[-1]) for l in text.splitlines() if l.startswith("# wall_seconds")), None)
    exit_code = next((int(l.split()[-1]) for l in text.splitlines() if l.startswith("# exit_code")), None)
    # pytest's final summary line, e.g. "2 failed, 14 passed, 34 skipped, 3 warnings in 1495.35s (0:24:55)"
    summary = None
    for line in reversed(text.splitlines()):
        s = line.strip("= ").strip()
        if re.search(r"\bin \d+(\.\d+)?s\b", s) and re.search(r"\d+ (passed|failed|skipped|error|deselected)", s):
            summary = s
            break
    counts = {k: 0 for k in COUNT_KEYS}
    seconds = None
    if summary:
        for n, word in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected|warnings?)",
                                  summary):
            key = {"error": "errors", "warning": "warnings"}.get(word, word)
            counts[key] = int(n)
        m = re.search(r"\bin (\d+(?:\.\d+)?)s\b", summary)
        seconds = float(m.group(1)) if m else None
    timeboxed = any(l.startswith("# TIMEBOX: sent SIGINT") for l in text.splitlines()) or "KeyboardInterrupt" in text
    timeouts = sorted(set(re.findall(r"^_{3,} (test_\S+\[[^\]]*\]) _{3,}$", text, flags=re.M)))
    timeout_hits = text.count("Failed: Timeout (>")
    return {"command": cmd, "cwd": cwd, "summary_line": summary, "pytest_seconds": seconds,
            "wall_seconds": wall, "exit_code": exit_code, "interrupted": timeboxed,
            "failed_tests": timeouts, "pytest_timeout_failures": timeout_hits, **counts}


def run_repro(sha):
    t0 = time.time()
    p = subprocess.run([str(ROOT / "scripts/harness/in_tree.sh"), sha[:8], PY, str(REPRO)],
                       capture_output=True, text=True, cwd="/tmp")
    secs = round(time.time() - t0, 2)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    try:
        js = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        js = None
    return {"stdout_json": js, "exit_code": p.returncode, "seconds": secs,
            "command": f"scripts/harness/in_tree.sh {sha[:8]} .venv/bin/python harness_only/repros/P1_tvd.py",
            "stdout_lines": len(lines)}


def main():
    buggy = run_repro(BUGGY)
    fixed = run_repro(FIXED)
    library = []
    for spec in LIBRARY_LOGS:
        path = LOGS / spec["log"]
        if not path.exists():
            library.append({"commit": spec["commit"], "files": spec["files"], "command": None, "passed": None,
                            "failed": None, "skipped": None, "errors": None, "xfailed": None, "seconds": None,
                            "subset_notes": spec["subset_notes"] + " [log missing: run did not complete]",
                            "log": str(path.relative_to(ROOT))})
            continue
        r = parse_log(path)
        note = spec["subset_notes"]
        if "full_interrupted" not in spec["log"]:
            if r["interrupted"]:
                note += " INTERRUPTED at the timebox: counts are pytest's summary at interruption (partial)."
            elif r["summary_line"]:
                note += " Completed within the timebox."
        if r["failed_tests"]:
            note += (f" Failed tests: {r['failed_tests']}; of the failures, {r['pytest_timeout_failures']} are "
                     f"pytest-timeout (>600 s).")
        library.append({
            "commit": spec["commit"],
            "files": spec["files"],
            "command": (f"cd {r['cwd']} && " if r["cwd"] else "") + (r["command"] or ""),
            "passed": r["passed"],
            "failed": r["failed"],
            "skipped": r["skipped"],
            "errors": r["errors"],
            "xfailed": r["xfailed"],
            "deselected": r["deselected"],
            "seconds": r["pytest_seconds"],
            "wall_seconds": r["wall_seconds"],
            "exit_code": r["exit_code"],
            "summary_line": r["summary_line"],
            "interrupted": r["interrupted"],
            "subset_notes": note,
            "log": str(path.relative_to(ROOT)),
        })
    bj, fj = buggy["stdout_json"] or {}, fixed["stdout_json"] or {}
    reproduces = buggy["exit_code"] == 1 and bj.get("bug_manifest") is True and fixed["exit_code"] == 0 \
        and fj.get("bug_manifest") is False
    notes = (
        "Repro: p = softmax(randn(8,512)) fp32; q = p.clone() (full tie) or half the columns copied from p; all 4 "
        "reductions. Buggy: every tied element gets -0.5*scale (e.g. -0.0625 = -0.5/8 for batchmean, matching the "
        "issue's H100 number); fixed: exactly 0. Loss identical at both commits (rel err ~1e-7). Backend: Triton "
        "CPU interpreter (TRITON_INTERPRET=1), torch 2.15.0 CPU, triton 3.9.0. Library tests: bf16 cases skip on "
        "CPU; the pre-existing suite cannot detect the bug (random inputs never tie); the fixed commit's tie tests "
        "fail against the buggy source (cross-check entry). See harness_only/phase1/tvd_notes.md."
    )
    out = {
        "pair_id": "P1_tvd",
        "buggy": {k: buggy[k] for k in ("stdout_json", "exit_code", "seconds", "command")},
        "fixed": {k: fixed[k] for k in ("stdout_json", "exit_code", "seconds", "command")},
        "reproduces_on_cpu": bool(reproduces),
        "library_tests": library,
        "notes": notes,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"pair_id": "P1_tvd", "reproduces_on_cpu": out["reproduces_on_cpu"],
                      "buggy_exit": buggy["exit_code"], "fixed_exit": fixed["exit_code"],
                      "library": [(l["commit"][:8], l["files"][:40], l.get("summary_line")) for l in library]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
