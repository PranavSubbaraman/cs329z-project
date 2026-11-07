"""Run the agent (or the static-review / no-Z3 conditions) on one benchmark instance, score it, and log it.

    python scripts/run_agent.py --instance tvd@39adfeef                 # full agent
    python scripts/run_agent.py --instance tvd@559d33e7 --condition no_z3
    python scripts/run_agent.py --instance ce@8c7a5078 --condition static --runs 3

Needs OPENROUTER_API_KEY (env or .env) and the exported tree (python scripts/export_trees.py). Appends one line per run to
results/runs.jsonl and saves the full trace to results/traces/<run_id>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.loop import run_agent  # noqa: E402
from agent.score import score  # noqa: E402
from scripts.env_report import env_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--condition", default="full", choices=["full", "no_z3", "static"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--max-calls", type=int, default=25)
    ap.add_argument("--max-seconds", type=float, default=600)
    a = ap.parse_args()
    inst = {i["instance_id"]: i for i in json.loads((ROOT / "benchmark" / "instances.json").read_text())["instances"]}[a.instance]
    for k in range(a.runs):
        print(f"=== {a.instance} condition={a.condition} run {k + 1}/{a.runs}")
        rec = run_agent(inst, condition=a.condition, max_calls=a.max_calls, max_seconds=a.max_seconds, run_tag=a.tag)
        rec["run_index"] = k
        rec["env"] = env_record()
        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not a.no_score:
            rec["score"] = score(rec)
        with open(ROOT / "results" / "runs.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        v = rec["verdict"]
        print(json.dumps({"verdict": v.get("verdict"), "location": v.get("location"), "proof_status": v.get("proof_status"),
                          "score": rec.get("score", {}).get("outcome"), "proof": rec.get("score", {}).get("proof"),
                          "localized": rec.get("score", {}).get("localized"), "tool_calls": rec["tool_calls"],
                          "wall_s": rec["wall_seconds"], "gpu_s": rec["gpu_seconds"], "cost_usd": rec["usage"]["cost_usd"],
                          "providers": rec["usage"]["providers"], "trace": rec["trace"]}, indent=1))


if __name__ == "__main__":
    main()
