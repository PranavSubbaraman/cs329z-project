"""HARNESS-ONLY driver: run harness_only/repros/P4_fljsd_bf16.py on the buggy and the fixed tree (via in_tree.sh),
record stdout JSON, exit code and wall seconds for each, and save them to
harness_only/phase1/logs/P4_fljsd_bf16_repro_runs.json (merged into P4_fljsd_bf16.json afterwards).

    .venv/bin/python harness_only/phase1/run_repro_P4_fljsd_bf16.py [buggy|fixed|both]
Runs the two trees concurrently when "both" (default). Extra env (e.g. TRITON_INTERPRET=0 on a GPU) passes through.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(ROOT, "harness_only", "phase1", "logs")
TREES = {"buggy": "5b828eef2bd2a723d001a7557b2e12da1817b505", "fixed": "ed1f856d5b00d4ff75a62d66cdb78a4ad935d02f"}
REPRO = os.path.join(ROOT, "harness_only", "repros", "P4_fljsd_bf16.py")


def run(which):
    sha = TREES[which]
    cmd = [os.path.join(ROOT, "scripts", "harness", "in_tree.sh"), sha[:8], os.path.join(ROOT, ".venv", "bin", "python"),
           REPRO]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    secs = time.time() - t0
    with open(os.path.join(LOGS, f"P4_fljsd_bf16_repro_{which}.stderr.log"), "w") as f:
        f.write(p.stderr)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    try:
        parsed = json.loads(lines[-1]) if len(lines) == 1 else {"unparsed_stdout": p.stdout}
    except json.JSONDecodeError:
        parsed = {"unparsed_stdout": p.stdout}
    return which, {"commit": sha, "command": " ".join(cmd[:2] + [".venv/bin/python", "harness_only/repros/P4_fljsd_bf16.py"]),
                   "stdout_json": parsed, "exit_code": p.returncode, "seconds": round(secs, 1),
                   "env": {k: os.environ.get(k) for k in ("TRITON_INTERPRET", "OMP_NUM_THREADS")}}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    names = ["buggy", "fixed"] if which == "both" else [which]
    path = os.path.join(LOGS, "P4_fljsd_bf16_repro_runs.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        for name, res in ex.map(run, names):
            results[name] = res
            print(name, res["exit_code"], res["seconds"], flush=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
