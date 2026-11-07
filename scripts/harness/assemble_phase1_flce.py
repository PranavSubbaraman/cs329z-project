"""HARNESS-ONLY: build harness_only/phase1/{P2_flce_strides,P3_flce_none}.json.

For each flce pair: runs its repro on the buggy and the fixed tree via scripts/harness/in_tree.sh (timed; stdout
JSON and exit code recorded), also runs it on the other pair's two commits as a cross-check, parses the Liger
library-test logs written by scripts/harness/run_liger_tests_flce.sh, and writes the result file.
Usage: .venv/bin/python scripts/harness/assemble_phase1_flce.py
"""
import json
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
IN_TREE = ROOT / "scripts" / "harness" / "in_tree.sh"
LOGS = ROOT / "harness_only" / "phase1" / "logs"
OUTDIR = ROOT / "harness_only" / "phase1"

SHAS = {
    "e44a5fe8": "e44a5fe86d3ee00def43913329e4c4ff70c3b416",
    "9ac26bb8": "9ac26bb82226c9279f0fda784300f5e02ce444b2",
    "395d6166": "395d6166e86f64f7c27f3c6e49cef89904add1aa",
    "ae47958e": "ae47958e39568874e3b21419130582a271f2a903",
}
PAIRS = {
    "P2_flce_strides": dict(buggy="395d6166", fixed="ae47958e", other=["e44a5fe8", "9ac26bb8"]),
    "P3_flce_none": dict(buggy="e44a5fe8", fixed="9ac26bb8", other=["395d6166", "ae47958e"]),
}


def run_repro(pair_id, prefix):
    cmd = [str(IN_TREE), prefix, str(PY), str(ROOT / "harness_only" / "repros" / f"{pair_id}.py")]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = round(time.time() - t0, 2)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip().startswith("{")]
    try:
        out = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        out = None
    return {"stdout_json": out, "exit_code": p.returncode, "seconds": dt,
            "command": "scripts/harness/in_tree.sh " + prefix + " .venv/bin/python harness_only/repros/" + pair_id + ".py"}


def parse_log(path):
    txt = path.read_text(errors="replace")
    cwd = re.search(r"^# cwd: (.*)$", txt, re.M)
    cmd = re.search(r"^# cmd: (.*)$", txt, re.M)
    wall = re.search(r"^WALL (\d+)", txt, re.M)
    summ = None
    for ln in txt.splitlines():
        if re.search(r"\b(passed|failed|skipped|error|errors|no tests ran)\b.* in [0-9.]+s", ln):
            summ = ln.strip("= ").strip()
    counts = {k: 0 for k in ("passed", "failed", "skipped", "errors", "xfailed")}
    secs = None
    if summ:
        for n, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|warnings?)", summ):
            k = "errors" if k.startswith("error") else k
            if k in counts:
                counts[k] = int(n)
        m = re.search(r" in ([0-9.]+)s", summ)
        secs = float(m.group(1)) if m else None
    failed_ids = re.findall(r"^FAILED (\S+)", txt, re.M)
    exc = {}
    for ln in re.findall(r"^E +((?:triton\.runtime\.errors\.)?[A-Za-z]+(?:Error|Exception)\b.*)$", txt, re.M):
        if ln.startswith("triton.runtime.errors.InterpreterError"):
            exc[ln[:160]] = exc.get(ln[:160], 0) + 1
    timeboxed = "TIMEBOX EXCEEDED" in txt
    return dict(cwd=cwd.group(1) if cwd else None, command=cmd.group(1) if cmd else None, summary=summ,
                counts=counts, seconds=secs, wall=int(wall.group(1)) if wall else None, failed_ids=failed_ids,
                interpreter_errors=exc, timeboxed=timeboxed)


def failure_note(info):
    if not info["failed_ids"]:
        return "no failures"
    n = len(info["failed_ids"])
    softcap = [i for i in info["failed_ids"] if "30.0" in i or "test_correctness_functional" in i]
    errs = "; ".join(f"{v}x {k}" for k, v in info["interpreter_errors"].items())
    if len(softcap) == n:
        return (f"all {n} failures are softcap=30.0 parametrizations (test_correctness / "
                f"test_correctness_with_forward_only second param set, which also sets ce_weight, and every "
                f"test_correctness_functional case) crashing in the Triton CPU interpreter: {errs}. "
                f"Interpreter limitation, not a numeric mismatch.")
    return f"{n} failures, {len(softcap)} softcap-parametrized; interpreter errors: {errs}"


def lib_entries(prefix):
    out = []
    for tag, f in (("transformers", "test/transformers/test_fused_linear_cross_entropy.py"),
                   ("ops", "test/ops/test_fused_linear_cross_entropy.py")):
        p = LOGS / f"flce_libtests_{prefix}_{tag}.log"
        if not p.exists():
            continue
        info = parse_log(p)
        note = "full file, no subset."
        if tag == "ops":
            note += " Every test is skipped: test/ops/conftest.py autouse fixture skips when CUDA is unavailable."
        else:
            note += " " + failure_note(info)
        if info["timeboxed"]:
            note += " TIMEBOX EXCEEDED (killed at 30 min)."
        out.append(dict(commit=SHAS[prefix], files=f, command=f"cd {info['cwd']} && {info['command']}",
                        **info["counts"], seconds=info["seconds"], subset_notes=note,
                        pytest_summary=info["summary"], wall_seconds=info["wall"], log=str(p.relative_to(ROOT))))
    return out


def transplant_entry():
    p = LOGS / "flce_libtests_transplant_P3_e44a5fe8src_9ac26bb8tests.log"
    if not p.exists():
        return []
    info = parse_log(p)
    return [dict(commit=SHAS["e44a5fe8"],
                 files="test/transformers/test_fused_linear_cross_entropy.py of 9ac26bb8 (fixed) run against "
                       "e44a5fe8 (buggy) src",
                 command=f"cd {info['cwd']} && {info['command']}", **info["counts"], seconds=info["seconds"],
                 subset_notes="TRANSPLANT: the fix PR's new regression test "
                              "(-k test_correctness_reduction_none_per_token_grad_output, 4 cases) collected from "
                              "the fixed tree but importing liger_kernel from the buggy tree's src (the tree's "
                              "pyproject sets pytest pythonpath=['src','.'], so -o 'pythonpath=<e44a5fe8>/src .' "
                              "is required; a helper plugin printed the imported path to confirm). Shows the "
                              "regression test catches the bug; this test does not exist at the buggy commit.",
                 pytest_summary=info["summary"], wall_seconds=info["wall"], log=str(p.relative_to(ROOT)))]


NOTES = {
    "P2_flce_strides": (
        "Reproduces on the Triton CPU interpreter. Repro: reduction='mean', input and weight are transposed leaves "
        "(stride (1, BT) / (1, V)), (0.5*loss).backward(); grad_input and grad_weight have max-normalized error 1.0 "
        "at 395d6166 (elements left unscaled or scaled twice by element_mul_kernel) and ~1.6e-7 at ae47958e; the "
        "contiguous-layout control agrees at both. Cross-check: the same repro also fires at e44a5fe8 and "
        "9ac26bb8 (both July commits predate the fix), and does not depend on reduction='none'. "
        "Liger's own suite does not exercise non-contiguous layouts with an upstream gradient != 1, so it passes "
        "the same tests at buggy and fixed (the fix PR added no tests); its 48 failures at both commits are "
        "softcap/ce_weight cases that crash in the interpreter. See harness_only/phase1/flce_notes.md."
    ),
    "P3_flce_none": (
        "Reproduces on the Triton CPU interpreter. Repro: contiguous float32, reduction='none', "
        "loss.backward(g) with non-uniform g; grad_input/grad_weight/grad_bias max-normalized errors ~0.4 at "
        "e44a5fe8 (all scaled by g[0]) and ~3e-7 at 9ac26bb8; the uniform-upstream control agrees at both. "
        "Contiguous layouts only, so the column-stride bug of P2 (present at both P3 commits) cannot fire. "
        "Cross-check: no manifestation at 395d6166/ae47958e (fix present). Liger's own suite at e44a5fe8 only "
        "backpropagates ones, so it passes the same non-softcap tests as at 9ac26bb8; the fix's 4 new regression "
        "tests (present only at 9ac26bb8) fail when transplanted onto e44a5fe8's src. 48 failures at each commit "
        "are softcap/ce_weight cases that crash in the interpreter. See harness_only/phase1/flce_notes.md."
    ),
}


def main():
    for pair_id, spec in PAIRS.items():
        buggy = run_repro(pair_id, spec["buggy"])
        fixed = run_repro(pair_id, spec["fixed"])
        cross = {SHAS[o]: run_repro(pair_id, o) for o in spec["other"]}
        rep = (buggy["exit_code"] == 1 and bool(buggy["stdout_json"]) and buggy["stdout_json"].get("bug_manifest")
               is True and fixed["exit_code"] == 0)
        lib = lib_entries(spec["buggy"]) + lib_entries(spec["fixed"])
        if pair_id == "P3_flce_none":
            lib += transplant_entry()
        res = {"pair_id": pair_id, "buggy": buggy, "fixed": fixed, "reproduces_on_cpu": bool(rep),
               "library_tests": lib, "notes": NOTES[pair_id], "cross_check_other_pair_commits": cross}
        (OUTDIR / f"{pair_id}.json").write_text(json.dumps(res, indent=1))
        print(pair_id, "reproduces_on_cpu=", rep, "buggy rc", buggy["exit_code"], "fixed rc", fixed["exit_code"],
              {k[:8]: v["exit_code"] for k, v in cross.items()})


if __name__ == "__main__":
    main()
