"""Tools available to the agent (OpenAI function-calling format) and their sandboxed implementations."""

from __future__ import annotations

import json
import textwrap

from .sandbox import Sandbox

MAX_OUT = 12000  # characters of tool output returned to the model

SPEC_DOC = ("An input spec is JSON: {\"tensors\": {<arg>: {\"shape\", \"dtype\", \"init\", \"edits\", \"requires_grad\", "
            "\"layout\"}}, \"kwargs\": {...}, \"upstream\": {...}, \"seed\": int}. See oracle/inputs.py for init kinds "
            "(randn, uniform, softmax, log_softmax, constant, values, copy, randint), edits (exact element/row/column "
            "overrides, incl. copying from another tensor), layouts (contiguous, transposed, row_strided:K, col_strided:K, "
            "offset:K) and upstream gradients (ones, scalar, randn, values, zeros). oracle.example_spec('<kernel>') gives a start. "
            "Example: {\"tensors\": {\"x\": {\"shape\": [4, 8], \"dtype\": \"float32\", \"init\": {\"kind\": \"randn\", \"std\": 2.0}, "
            "\"edits\": [{\"index\": [0, 3], \"value\": 0.0}, {\"rows\": [1], \"value\": 1.0}], \"layout\": \"transposed\"}}, "
            "\"kwargs\": {\"reduction\": \"sum\"}, \"upstream\": {\"kind\": \"scalar\", \"value\": 0.5}, \"seed\": 0} "
            "(\"from\" must name a tensor listed earlier in \"tensors\").")

TOOLS = {
    "list_files": {
        "description": "List a directory inside the sandbox (paths are relative to the sandbox root: liger/, task/, oracle/, scratch/).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}, "required": []},
    },
    "read_file": {
        "description": "Read a text file inside the sandbox, with line numbers. Use start_line/end_line for long files (max 400 lines per call).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"},
                                                         "end_line": {"type": "integer"}}, "required": ["path"]},
    },
    "run_gradient_test": {
        "description": ("Execute a Python snippet in the sandbox (fresh process, cwd scratch/, no network, timeout 180 s). "
                        "The Liger tree is importable (import liger_kernel), and so are the oracle helpers: from oracle import "
                        "grad_compare, build_inputs, load_task, diff_args, replay, check_properties, example_spec, sample. "
                        "kernel, reference = load_task() gives the entry point and the float64-oracle reference. Print what you need; "
                        "stdout/stderr are returned (truncated)."),
        "parameters": {"type": "object", "properties": {"python_snippet": {"type": "string"}}, "required": ["python_snippet"]},
    },
    "check_properties": {
        "description": ("Run the property checks on one input spec: zero_upstream, layout_invariance, no_mutation_without_grad, "
                        "compile_matches_eager, determinism. " + SPEC_DOC),
        "parameters": {"type": "object", "properties": {
            "entry_point": {"type": "string", "default": "entry.py:kernel"},
            "input_spec": {"type": "object"},
            "properties": {"type": "array", "items": {"type": "string"}}}, "required": ["input_spec"]},
    },
    "z3_check": {
        "description": ("Run a Z3 Python script (import z3 is available; timeout 60 s). The script should model (a) the kernel's "
                        "backward as read from the source and (b) the reference derivative, assert that they differ, and leave a "
                        "z3.Solver in a variable named `solver` (or call oracle.z3tools.report(solver)). Returns sat with a model "
                        "(a counterexample to equality), unsat (equal on the encoded domain), or unknown (timeout). exp and log may "
                        "be left uninterpreted (z3.Function)."),
        "parameters": {"type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"]},
    },
    "conformance_check": {
        "description": ("Check that your model of the kernel behaves like the real kernel: model_fn is Python source defining "
                        "`def model(inputs: dict) -> dict`. inputs holds float64 copies of every tensor argument, the kwargs, and "
                        "'upstream' (the float64 upstream gradient). It returns {'grad_<arg>': tensor, ...} (optionally 'output'). "
                        "The model is compared with the kernel under the oracle's tolerance rule on random in-domain inputs, "
                        "including generic special values (shared entries across inputs, zeroed ranges). A proof counts only if "
                        "this passes."),
        "parameters": {"type": "object", "properties": {"model_fn": {"type": "string"},
                                                         "entry_point": {"type": "string", "default": "entry.py:kernel"},
                                                         "n_trials": {"type": "integer", "default": 24}},
                       "required": ["model_fn"]},
    },
    "replay": {
        "description": ("Replay a concrete counterexample: build the tensors from an input spec, run grad_compare (kernel in its "
                        "native dtype vs the float64 reference) and the property checks, and report whether a violation "
                        "reproduces. A bug claim must include a spec that replays. " + SPEC_DOC),
        "parameters": {"type": "object", "properties": {"assignment": {"type": "object", "description":
                       "{\"input_spec\": <spec>, \"checks\": optional list among grad_compare and the property names}"}},
                       "required": ["assignment"]},
    },
    "submit_verdict": {
        "description": "Finish the task. Call exactly once.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": ["bug", "no_bug", "inconclusive"]},
            "counterexample": {"type": ["object", "null"], "description": "input spec that replays (required for bug)"},
            "location": {"type": "string", "description": "file:line of the defect, e.g. liger/src/liger_kernel/ops/x.py:123"},
            "evidence": {"type": "string"},
            "proof_status": {"type": "string", "enum": ["none", "sat", "unsat_with_conformance", "unsat_without_conformance", "unknown"]}},
            "required": ["verdict", "evidence", "proof_status"]},
    },
}

FULL = ["list_files", "read_file", "run_gradient_test", "check_properties", "z3_check", "conformance_check", "replay", "submit_verdict"]
NO_Z3 = [t for t in FULL if t != "z3_check"]
STATIC = ["list_files", "read_file", "submit_verdict"]


def schemas(names):
    return [{"type": "function", "function": {"name": n, **TOOLS[n]}} for n in names]


def _trim(s: str, n: int = MAX_OUT) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + f"\n... [{len(s) - n} chars truncated] ...\n" + s[-n // 2:]


class ToolRunner:
    def __init__(self, sandbox: Sandbox):
        self.sb = sandbox

    def _run_json(self, body: str, args: dict, timeout: float, label: str) -> dict:
        (self.sb.dir / "scratch" / "_args.json").write_text(json.dumps(args))
        code = ("import json\nfrom oracle import *\nfrom oracle.kernels import load_task, kernel_name\n"
                "ARGS = json.loads(open('_args.json').read())\n" + textwrap.dedent(body))
        r = self.sb.run_python(code, timeout=timeout, label=label)
        res = None
        for line in r["stdout"].splitlines()[::-1]:
            if line.startswith("RESULT "):
                try:
                    res = json.loads(line[7:])
                except Exception:
                    pass
                break
        return {"result": res, "returncode": r["returncode"], "timed_out": r["timed_out"], "seconds": r["seconds"],
                "stderr_tail": _trim(r["stderr"], 3000) if (res is None or r["returncode"]) else ""}

    def call(self, name: str, a: dict) -> str:
        try:
            out = getattr(self, "t_" + name)(**a)
        except TypeError as e:
            out = {"error": f"bad arguments for {name}: {e}"}
        except PermissionError as e:
            out = {"error": str(e)}
        except Exception as e:  # tool infrastructure error
            out = {"error": f"{type(e).__name__}: {e}"}
        return out if isinstance(out, str) else _trim(json.dumps(out, default=str, indent=1))

    # ---- tools ---------------------------------------------------------------------------------------------------
    def t_list_files(self, path="."):
        p = self.sb.resolve(path)
        if not p.is_dir():
            return {"error": f"not a directory: {path}"}
        ents = []
        for c in sorted(p.iterdir()):
            if c.name in ("_boot",) or c.name.startswith("_args"):
                continue
            ents.append(c.name + ("/" if c.is_dir() else f"  ({c.stat().st_size} B)"))
        return "\n".join(ents[:400]) + ("\n... (truncated)" if len(ents) > 400 else "")

    def t_read_file(self, path, start_line=1, end_line=None):
        p = self.sb.resolve(path)
        if not p.is_file():
            return {"error": f"not a file: {path}"}
        lines = p.read_text(errors="replace").splitlines()
        s = max(1, int(start_line or 1))
        e = min(len(lines), int(end_line) if end_line else s + 399, s + 399)
        body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(s, e + 1))
        return f"{path} (lines {s}-{e} of {len(lines)})\n" + body

    def t_run_gradient_test(self, python_snippet):
        r = self.sb.run_python(python_snippet, timeout=180, label="test")
        return {"returncode": r["returncode"], "timed_out": r["timed_out"], "seconds": r["seconds"],
                "stdout": _trim(r["stdout"], 9000), "stderr": _trim(r["stderr"], 3000)}

    def t_check_properties(self, input_spec, entry_point="entry.py:kernel", properties=None):
        body = """
        kernel, reference = load_task(entry=ARGS['entry_point'])
        r = check_properties(kernel, reference, ARGS['input_spec'], kernel_name(), which=ARGS.get('properties'))
        print('RESULT ' + json.dumps(r, default=str))
        """
        return self._run_json(body, {"input_spec": input_spec, "entry_point": entry_point, "properties": properties}, 300, "props")

    def t_z3_check(self, script):
        wrapped = (script + "\n\n# --- harness footer ---\n"
                   "import json as _j\nfrom oracle.z3tools import solve as _solve\n"
                   "if 'solver' in globals() and not globals().get('_KV_REPORTED'):\n"
                   "    print('Z3_RESULT ' + _j.dumps(_solve(solver, 30000)))\n")
        r = self.sb.run_python(wrapped, timeout=60, label="z3")
        res = None
        for line in r["stdout"].splitlines()[::-1]:
            if line.startswith("Z3_RESULT "):
                res = json.loads(line[len("Z3_RESULT "):])
                break
        if res is None:
            res = {"result": "unknown", "reason": "timeout" if r["timed_out"] else "no solver result (define `solver`)"}
        return {"z3": res, "stdout": _trim(r["stdout"], 3000), "stderr": _trim(r["stderr"], 2000), "seconds": r["seconds"]}

    def t_conformance_check(self, model_fn, entry_point="entry.py:kernel", n_trials=24):
        body = """
        kernel, reference = load_task(entry=ARGS['entry_point'])
        r = conformance_check(ARGS['model_fn'], kernel, reference, kernel_name(), n_trials=int(ARGS['n_trials']))
        print('RESULT ' + json.dumps(r, default=str))
        """
        return self._run_json(body, {"model_fn": model_fn, "entry_point": entry_point, "n_trials": min(int(n_trials), 48)}, 300, "conform")

    def t_replay(self, assignment):
        body = """
        r = replay(ARGS['assignment'])
        gc = r.get('grad_compare', {})
        r['grad_compare'] = {'pass': gc.get('pass'), 'error': gc.get('error'), 'failed': gc.get('failed'),
                             'items': [{k: it.get(k) for k in ('name','dtype','shape','max_abs_err','max_rel_err','allowed','ratio','passed','where','kernel_at','ref_at','reason')} for it in gc.get('items', [])]}
        print('RESULT ' + json.dumps(r, default=str))
        """
        return self._run_json(body, {"assignment": assignment}, 300, "replay")
