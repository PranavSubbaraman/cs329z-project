"""Harness-side scoring of one run record (identical for every method; see PREREGISTRATION.md).

Every bug claim is replayed with the trusted oracle on the instance's own exported tree (scripts/harness/replay_claim.py,
in a subprocess). Labels and fix-diff lines come from harness_only/labels.json and never reach the agent.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def labels():
    pairs = json.loads((ROOT / "harness_only" / "labels.json").read_text())["pairs"]
    side = {}
    for p in pairs:
        side[p["buggy"]] = (p, "buggy")
        if p.get("fixed"):
            side[p["fixed"]] = (p, "fixed")
    return side


def trusted_replay(kernel: str, commit: str, spec: dict, timeout: float = 600) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        path = f.name
    env = {**os.environ, "KV_KERNEL": kernel, "KV_TASK_DIR": str(ROOT / "benchmark" / "kernels" / kernel)}
    py = os.environ.get("KV_PY", str(ROOT / (".venv-gpu" if (ROOT / ".venv-gpu").exists() else ".venv") / "bin" / "python"))
    try:
        p = subprocess.run([str(ROOT / "scripts" / "harness" / "in_tree.sh"), commit[:8], py,
                            str(ROOT / "scripts" / "harness" / "replay_claim.py"), path],
                           capture_output=True, text=True, timeout=timeout, env=env, cwd=ROOT)
        for line in p.stdout.splitlines()[::-1]:
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        return {"reproduces": False, "replay_error": (p.stderr or p.stdout)[-1500:]}
    except subprocess.TimeoutExpired:
        return {"reproduces": False, "replay_error": "trusted replay timed out"}
    finally:
        os.unlink(path)


def _proof_evidence(trace_path: str | None) -> dict:
    ev = {"z3_unsat": False, "conformance_pass": False}
    if not trace_path or not (ROOT / trace_path).exists():
        return ev
    for e in json.loads((ROOT / trace_path).read_text()).get("events", []):
        if e.get("type") != "tool":
            continue
        try:
            r = json.loads(e.get("result") or "{}")
        except Exception:
            continue
        if e["name"] == "z3_check" and (r.get("z3") or {}).get("result") == "unsat":
            ev["z3_unsat"] = True
        if e["name"] == "conformance_check" and (r.get("result") or {}).get("conformance") == "pass":
            ev["conformance_pass"] = True
    return ev


def _z3_sat_in_trace(trace_path):
    if not trace_path or not (ROOT / trace_path).exists():
        return False
    for e in json.loads((ROOT / trace_path).read_text()).get("events", []):
        if e.get("type") == "tool" and e.get("name") == "z3_check":
            try:
                if (json.loads(e.get("result") or "{}").get("z3") or {}).get("result") == "sat":
                    return True
            except Exception:
                pass
    return False


def localized(location: str | None, pair: dict) -> bool | None:
    if not location:
        return False
    m = re.search(r"(src/liger_kernel/[\w/.\-]+\.py):(\d+)", location)
    if not m:
        return False
    f, line = m.group(1), int(m.group(2))
    return any(lo <= line <= hi for lo, hi in pair.get("fix_lines", {}).get(f, []))


def score(record: dict) -> dict:
    pair, side = labels()[record["commit"]]
    v = record["verdict"] or {}
    verdict = v.get("verdict", "inconclusive")
    out = {"pair_id": pair["pair_id"], "bug_class": pair["bug_class"], "side": side, "verdict": verdict}
    if verdict == "bug":
        ce = v.get("counterexample")
        rep = trusted_replay(record["kernel"], record["commit"], ce) if isinstance(ce, dict) and ce else \
            {"reproduces": False, "replay_error": "no counterexample given"}
        out["replay"] = rep
        replays = bool(rep.get("reproduces"))
        if side == "buggy":
            out["outcome"] = "detection" if replays else "miss_unconfirmed_claim"
            out["localized"] = localized(v.get("location"), pair) if replays else False
        else:
            out["outcome"] = "possible_new_bug" if replays else "false_alarm"
            if replays:
                out["known_other_bugs_here"] = pair.get("known_other_bugs", {}).get(record["commit"][:8], [])
    elif verdict == "no_bug":
        out["outcome"] = "miss" if side == "buggy" else "correct_no_bug"
    else:
        out["outcome"] = "miss" if side == "buggy" else "inconclusive"
    ps = v.get("proof_status")
    if ps == "sat":
        out["sat_supported_by_trace"] = _z3_sat_in_trace(record.get("trace"))
    if verdict == "no_bug" and ps == "unsat_with_conformance":
        ev = _proof_evidence(record.get("trace"))
        out["proof_evidence"] = ev
        if ev["z3_unsat"] and ev["conformance_pass"]:
            out["proof"] = "proof" if side == "fixed" else "false_proof"
        else:
            out["proof"] = "claimed_but_unsupported"
    return out
