"""replay(assignment): turn a concrete counterexample into tensors and report whether it reproduces a violation.

    assignment = {"input_spec": <spec, oracle.inputs format>, "checks": ["grad_compare", "zero_upstream", ...]}
    (a bare spec is accepted too; checks default to grad_compare plus every property)

A violation is a failed grad_compare (tolerance rule) or a failed property. A kernel crash is reported under
"errors" but is not a silent-gradient violation. Used by the agent's replay tool and by the harness when it scores a
claimed counterexample.
"""

from __future__ import annotations

import time

from .compare import grad_compare
from .inputs import build_inputs
from .kernels import kernel_name, load_task
from .properties import PROPERTIES, check_properties
from .spaces import diff_args


def replay(assignment: dict, kernel: str | None = None, task_directory=None, kernel_fn=None, ref_fn=None) -> dict:
    t0 = time.time()
    spec = assignment.get("input_spec", assignment) if isinstance(assignment, dict) else assignment
    checks = assignment.get("checks") if isinstance(assignment, dict) and "input_spec" in assignment else None
    checks = checks or ["grad_compare"] + PROPERTIES
    kernel = kernel or kernel_name()
    if kernel_fn is None or ref_fn is None:
        kernel_fn, ref_fn = load_task(task_directory)
    out = {"kernel": kernel, "checks": checks, "violations": [], "errors": []}
    if "grad_compare" in checks:
        try:
            gc = grad_compare(kernel_fn, ref_fn, build_inputs(spec, diff_args=diff_args(kernel)))
        except Exception as e:
            gc = {"pass": False, "error": f"could not build or run the spec: {type(e).__name__}: {e}"[:1000]}
        out["grad_compare"] = gc
        if gc.get("error"):
            out["errors"].append("grad_compare: " + gc["error"][:300])
        elif not gc.get("pass"):
            out["violations"].append("grad_compare")
    props = [c for c in checks if c in PROPERTIES]
    if props:
        pr = check_properties(kernel_fn, ref_fn, spec, kernel, which=props)
        out["properties"] = pr["results"]
        out["violations"] += pr["violations"]
        out["errors"] += [f"{p}: {r.get('reason', '')[:200]}" for p, r in pr["results"].items() if r["status"] == "error"]
    out["reproduces"] = bool(out["violations"])
    out["seconds"] = round(time.time() - t0, 3)
    return out
