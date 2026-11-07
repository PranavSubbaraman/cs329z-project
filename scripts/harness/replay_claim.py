"""HARNESS-ONLY: replay a claimed counterexample with the trusted oracle on one tree (run via in_tree.sh with
KV_KERNEL and KV_TASK_DIR=benchmark/kernels/<kernel>). Prints one JSON line."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oracle import replay  # noqa: E402

spec = json.loads(Path(sys.argv[1]).read_text())
r = replay({"input_spec": spec})
gc = r.get("grad_compare", {})
print("RESULT " + json.dumps({"reproduces": r["reproduces"], "violations": r["violations"], "errors": r["errors"][:5],
                              "grad_failed": gc.get("failed"), "grad_error": gc.get("error")}, default=str))
