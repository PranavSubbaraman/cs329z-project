"""Print an agent trace (results/traces/<run_id>.json) compactly: reasoning text, tool calls, short results, verdict.
    python scripts/show_trace.py results/traces/<run_id>.json [--full]"""
import json
import sys
import textwrap

t = json.load(open(sys.argv[1]))
full = "--full" in sys.argv
r = t.get("record", {})
print(f"run {t['run_id']}  instance {t['instance_id']}  condition {t['condition']}  model {t['model']}  backend {t['backend']}")
n = 0
for e in t["events"]:
    if e["type"] == "assistant":
        if e.get("content"):
            print("\n\033[0;33mAGENT:\033[0m " + textwrap.shorten(e["content"], 600 if not full else 100000))
        for tc in e.get("tool_calls") or []:
            n += 1
            args = tc["function"]["arguments"]
            print(f"\033[1;36m[{n}] {tc['function']['name']}\033[0m " + (args if full else textwrap.shorten(args, 300)))
    elif e["type"] == "tool" and e["name"] != "submit_verdict":
        print("    -> " + (e["result"] if full else textwrap.shorten(e["result"], 260)))
v = r.get("verdict", {})
print(f"\n\033[1;32mVERDICT\033[0m {v.get('verdict')}  location={v.get('location')}  proof_status={v.get('proof_status')}")
print("evidence: " + textwrap.shorten(v.get("evidence") or "", 700))
print(f"tool calls {r.get('tool_calls')}  wall {r.get('wall_seconds')} s  GPU-s {r.get('gpu_seconds')}  cost ${r.get('usage', {}).get('cost_usd')}")
