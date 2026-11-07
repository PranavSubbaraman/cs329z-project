"""Helpers for Z3 checks of a kernel's backward against the reference derivative.

The z3_check tool runs a script that builds a z3.Solver named `solver` (or calls report(solver) itself). The usual
shape: declare inputs, encode (a) the kernel's backward as read from its source and (b) the reference derivative,
assert they differ, and check. SAT = a counterexample (the model lists the assignment); UNSAT = they are equal on
the encoded domain; unknown = timeout or incompleteness. exp/log may be left uninterpreted (z3.Function).
"""

from __future__ import annotations

import json

import z3


def model_to_dict(m) -> dict:
    out = {}
    for d in m.decls():
        v = m[d]
        try:
            if z3.is_rational_value(v):
                out[d.name()] = float(v.numerator_as_long()) / float(v.denominator_as_long())
                out[d.name() + "__exact"] = str(v)
            elif z3.is_int_value(v):
                out[d.name()] = v.as_long()
            elif z3.is_true(v) or z3.is_false(v):
                out[d.name()] = bool(z3.is_true(v))
            elif z3.is_fp_value(v):
                r = z3.simplify(z3.fpToReal(v))  # NaN/inf have no real value: falls through to str()
                out[d.name()] = (float(r.numerator_as_long()) / float(r.denominator_as_long())
                                 if z3.is_rational_value(r) else str(v))
            else:
                out[d.name()] = str(v)
        except Exception:
            out[d.name()] = str(v)
    return out


def solve(solver: "z3.Solver", timeout_ms: int = 30000) -> dict:
    solver.set("timeout", int(timeout_ms))
    r = solver.check()
    res = {"result": str(r)}  # "sat" | "unsat" | "unknown"
    if r == z3.sat:
        res["model"] = model_to_dict(solver.model())
    elif r == z3.unknown:
        res["reason"] = solver.reason_unknown()
    res["n_assertions"] = len(solver.assertions())
    return res


def report(solver: "z3.Solver", timeout_ms: int = 30000) -> dict:
    """Solve and print the result as one JSON line prefixed with Z3_RESULT (what the z3_check tool reads)."""
    res = solve(solver, timeout_ms)
    print("Z3_RESULT " + json.dumps(res), flush=True)
    return res
