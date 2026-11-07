"""Z3 check of the TVD backward against the reference derivative (the encoding the agent is expected to write).

Kernel backward as read from liger/src/liger_kernel/ops/tvd.py (per element, s = reduction scale > 0):
  buggy commit: grad = where(p > q,  0.5*s, -0.5*s)
  fixed commit: grad = where(p == q, 0, where(p > q, 0.5*s, -0.5*s))
Reference: d/dp 0.5*|p - q| * s = 0.5*s*sgn(p - q), with sgn(0) = 0 (torch.abs backward; the subgradient at the kink).
We assert they differ: SAT gives a counterexample, UNSAT proves equality on the encoded domain (all reals p, q, s > 0).
    python scripts/demo/z3_tvd.py buggy|fixed
"""
import json
import sys

import z3

variant = sys.argv[1] if len(sys.argv) > 1 else "buggy"
p, q, s = z3.Reals("p q s")
half = z3.RealVal("1/2")
if variant == "buggy":
    kernel_grad = z3.If(p > q, half * s, -half * s)
else:
    kernel_grad = z3.If(p == q, z3.RealVal(0), z3.If(p > q, half * s, -half * s))
sgn = z3.If(p > q, 1, z3.If(p < q, -1, 0))
ref_grad = half * s * sgn
solver = z3.Solver()
solver.add(s > 0, kernel_grad != ref_grad)
r = solver.check()
out = {"variant": variant, "result": str(r)}
if r == z3.sat:
    m = solver.model()
    out["counterexample"] = {str(v): str(m[v]) for v in (p, q, s)}
    out["kernel_grad"] = str(m.eval(kernel_grad))
    out["reference_grad"] = str(m.eval(ref_grad))
print(json.dumps(out))
