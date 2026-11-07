"""One-call OpenRouter smoke test with a dummy tool, before any agent loop exists.

Sends a single chat-completions request offering one dummy tool and checks that:
  - the pinned provider served it (response field, cross-checked against the /generation endpoint),
  - the model returned a well-formed tool call (valid JSON arguments matching the schema),
  - usage and cost were reported.
Writes results/smoke_openrouter.json.

    python scripts/smoke_openrouter.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.llm import OpenRouterClient, record_to_dict  # noqa: E402

DUMMY_TOOL = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two integers and return their sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}


def main():
    client = OpenRouterClient(max_tokens=256)
    messages = [{"role": "user", "content": "Use the add_numbers tool to add 17 and 25. Do not answer directly."}]
    resp, rec = client.chat(messages, tools=[DUMMY_TOOL], tool_choice="auto")
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    ok_call = False
    args = None
    if calls:
        fn = calls[0]["function"]
        args = json.loads(fn["arguments"])
        ok_call = fn["name"] == "add_numbers" and args == {"a": 17, "b": 25}
    time.sleep(1.0)
    gen = client.generation_stats(rec.generation_id) if rec.generation_id else None
    out = {
        "model_requested": client.model,
        "provider_prefs": client.provider_prefs(),
        "call": record_to_dict(rec),
        "response_provider_field": resp.get("provider"),
        "generation_endpoint": {k: (gen or {}).get(k) for k in
                                ["provider_name", "model", "total_cost", "tokens_prompt", "tokens_completion",
                                 "native_tokens_prompt", "native_tokens_completion", "native_tokens_reasoning",
                                 "latency", "generation_time", "finish_reason"]},
        "tool_call": calls[0] if calls else None,
        "tool_call_ok": ok_call,
        "assistant_content": msg.get("content"),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "smoke_openrouter.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    sys.exit(0 if ok_call else 1)


if __name__ == "__main__":
    main()
