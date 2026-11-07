"""ReAct-style agent loop over OpenRouter chat completions with tool calling, with budgets and full trace logging.

    from agent.loop import run_agent
    rec = run_agent(instance, condition="full")      # "full" | "no_z3" | "static"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .llm import OpenRouterClient, record_to_dict
from .prompts import AGENT, NO_Z3_NOTE, STATIC, user_message
from .sandbox import ROOT, Sandbox
from .tools import FULL, NO_Z3, STATIC as STATIC_TOOLS, ToolRunner, schemas

CONDITIONS = {"full": (AGENT, FULL), "no_z3": (AGENT + NO_Z3_NOTE, NO_Z3), "static": (STATIC, STATIC_TOOLS)}


def run_agent(instance: dict, condition: str = "full", max_calls: int = 25, max_seconds: float = 600.0,
              client: OpenRouterClient | None = None, run_tag: str = "", log=print, keep_sandbox: bool = False) -> dict:
    sysprompt, tool_names = CONDITIONS[condition]
    sysprompt = sysprompt.format(max_calls=max_calls, max_minutes=int(max_seconds // 60))
    client = client or OpenRouterClient()
    sb = Sandbox(instance)
    runner = ToolRunner(sb)
    task_md = (sb.dir / "task" / "task.md").read_text()
    messages = [{"role": "system", "content": sysprompt},
                {"role": "user", "content": user_message(task_md, instance["kernel"])}]
    tools = schemas(tool_names)
    t0 = time.time()
    calls, verdict, nudges, stop_reason = [], None, 0, None
    trace = {"run_id": sb.run_id, "instance_id": instance["instance_id"], "condition": condition, "model": client.model,
             "provider_prefs": client.provider_prefs(), "max_calls": max_calls, "max_seconds": max_seconds,
             "backend": sb.backend, "events": []}
    while verdict is None:
        elapsed = time.time() - t0
        if len(calls) >= max_calls:
            stop_reason = "tool-call budget exhausted"
            break
        if elapsed >= max_seconds:
            stop_reason = "wall-clock budget exhausted"
            break
        remaining = max_calls - len(calls)
        if remaining <= 3 and not any(m.get("_budget_note") for m in messages):
            messages.append({"role": "user", "content": f"Budget: {remaining} tool calls left. Submit your verdict soon.",
                             "_budget_note": True})
        try:
            resp, rec = client.chat([{k: v for k, v in m.items() if not k.startswith("_")} for m in messages], tools=tools)
        except Exception as e:
            stop_reason = f"LLM error: {type(e).__name__}: {e}"[:300]
            trace["events"].append({"type": "llm_error", "error": stop_reason, "t": round(time.time() - t0, 2)})
            break
        msg = resp["choices"][0]["message"]
        am = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            am["tool_calls"] = [{"id": tc["id"], "type": "function",
                                 "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                                for tc in msg["tool_calls"]]
        messages.append(am)
        trace["events"].append({"type": "assistant", "content": am["content"], "tool_calls": am.get("tool_calls"),
                                "llm": record_to_dict(rec), "t": round(time.time() - t0, 2)})
        if am["content"]:
            log(f"  [assistant] {am['content'][:300]}")
        if not am.get("tool_calls"):
            nudges += 1
            if nudges > 2:
                stop_reason = "model stopped calling tools"
                break
            messages.append({"role": "user", "content": "Continue: use the tools, and finish with submit_verdict."})
            continue
        for tc in am["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                args, out = None, json.dumps({"error": f"arguments are not valid JSON: {e}"})
            if len(calls) >= max_calls:
                out = json.dumps({"error": "tool-call budget exhausted"})
            elif args is not None and name not in tool_names:
                out = json.dumps({"error": f"unknown tool {name}"})
            elif args is not None and name == "submit_verdict":
                verdict = args
                out = json.dumps({"ok": True})
            elif args is not None:
                t1 = time.time()
                out = runner.call(name, args)
                log(f"  [tool {len(calls) + 1}] {name} ({time.time() - t1:.1f}s): {out[:160]!r}")
            calls.append({"name": name, "t": round(time.time() - t0, 2), "chars": len(out)})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            trace["events"].append({"type": "tool", "name": name, "arguments": args, "result": out,
                                    "t": round(time.time() - t0, 2)})
            if verdict is not None:
                break
    wall = time.time() - t0
    if verdict is None:
        verdict = {"verdict": "inconclusive", "counterexample": None, "location": None,
                   "evidence": f"forced: {stop_reason}", "proof_status": "none", "forced": True}
    record = {"run_id": sb.run_id, "run_tag": run_tag, "instance_id": instance["instance_id"], "kernel": instance["kernel"],
              "commit": instance["commit"], "condition": condition, "verdict": verdict, "stop_reason": stop_reason,
              "tool_calls": len(calls), "tool_counts": {n: sum(c["name"] == n for c in calls) for n in tool_names},
              "wall_seconds": round(wall, 2), "gpu_seconds": round(sb.exec_seconds, 2), "backend": sb.backend,
              "usage": client.usage.totals()}
    trace["record"] = record
    tdir = ROOT / "results" / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{sb.run_id}.json").write_text(json.dumps(trace, indent=1, default=str) + "\n")
    record["trace"] = f"results/traces/{sb.run_id}.json"
    if not keep_sandbox:
        sb.cleanup()
    return record
