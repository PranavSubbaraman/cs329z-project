"""Property checks (tolerance-free except where stated), applied to one input spec:

  zero_upstream            backward with a zero upstream gradient gives exactly-zero (and finite) input gradients
  layout_invariance        the same values in other memory layouts (transposed / strided / offset views of the inputs,
                           strided upstream gradients) give results that still satisfy the tolerance rule; checked only
                           when the all-contiguous baseline satisfies it
  no_mutation_without_grad with requires_grad=False on every input (plain call and under torch.no_grad()), no input tensor
                           changes; with gradients enabled, inputs that receive no gradient do not change either
                           (a write into an input that requires grad is reported but is not a violation: some kernels
                           store their gradient there by documented design)
  compile_matches_eager    torch.compile(kernel) satisfies the tolerance rule whenever eager does
  determinism              two identical runs give bitwise-identical outputs and gradients

    from oracle import check_properties
    res = check_properties(kernel, reference, spec, "tvd")      # res["violations"] lists failed properties
"""

from __future__ import annotations

import os
import time

import torch
import torch._dynamo

from .compare import _as_tuple, grad_compare
from .inputs import build_inputs, clone_like, make_upstream
from .spaces import ARGS, SCOPE_LAYOUTS, diff_args

PROPERTIES = ["zero_upstream", "layout_invariance", "no_mutation_without_grad", "compile_matches_eager", "determinism"]


def _fresh(built):
    return {k: clone_like(v) if isinstance(v, torch.Tensor) else v for k, v in built.call_args().items()}


def _run(kernel_fn, args, up, seed, zero=False):
    outs = _as_tuple(kernel_fn(**args))
    diff = [k for k, v in args.items() if isinstance(v, torch.Tensor) and v.requires_grad]
    grads = {}
    req = [o for o in outs if o.requires_grad]
    if diff and req:
        gs = [torch.zeros_like(o) if zero else make_upstream(up, o, o.dtype, seed) for o in req]
        gg = torch.autograd.grad(req, [args[k] for k in diff], gs, allow_unused=True)
        grads = dict(zip(diff, gg))
    return outs, grads


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(torch.nan_to_num(a.detach(), nan=7.77e7),
                                                                          torch.nan_to_num(b.detach(), nan=7.77e7)))


def prop_zero_upstream(kernel_fn, ref_fn, spec, kernel):
    built = build_inputs(spec, diff_args=diff_args(kernel))
    args = _fresh(built)
    if not any(isinstance(v, torch.Tensor) and v.requires_grad for v in args.values()):
        return {"status": "skip", "reason": "no input requires grad"}
    _, grads = _run(kernel_fn, args, built.upstream, 0, zero=True)
    bad = {}
    for k, g in grads.items():
        if g is None:
            continue
        gd = g.detach()
        nz = int((gd != 0).sum()) + int((~torch.isfinite(gd)).sum())
        if nz:
            bad[k] = {"nonzero_or_nonfinite": nz, "numel": gd.numel(), "max_abs": float(gd.abs().nan_to_num(0).max())}
    return {"status": "fail" if bad else "pass", "violations": bad}


def _variants(kernel, built):
    meta = ARGS[kernel]
    scope = [l for l in SCOPE_LAYOUTS[kernel] if l != "contiguous"]
    out = []
    names = [k for k, v in built.tensors.items() if isinstance(v, torch.Tensor) and v.dim() >= 1]
    for k in names:
        mat = meta.get(k, {}).get("matrix", False) and built.tensors[k].dim() == 2
        lay = scope[0] if mat else ("strided" if "offset:3" not in scope else "offset:3")
        out.append((f"{k}:{lay}", {k: lay}, None))
    combo = {k: (scope[0] if meta.get(k, {}).get("matrix") and built.tensors[k].dim() == 2 else "strided") for k in names}
    out.append(("all_non_contiguous", combo, None))
    up = dict(built.upstream)
    if up.get("kind") in ("randn", "values", "ones", "scalar"):
        out.append(("upstream:strided", {}, {**up, "layout": "strided"}))
    return out


def prop_layout_invariance(kernel_fn, ref_fn, spec, kernel):
    base_layouts = {k: "contiguous" for k in (spec.get("tensors") or {})}
    base_spec = {**spec, "upstream": {**(spec.get("upstream") or {"kind": "ones"}), "layout": "contiguous"}}
    base = build_inputs(base_spec, diff_args=diff_args(kernel), layout_override=base_layouts)
    rb = grad_compare(kernel_fn, ref_fn, base)
    if not rb.get("pass"):
        return {"status": "inconclusive", "reason": "contiguous baseline already fails the tolerance rule",
                "baseline_failed": rb.get("failed"), "baseline_error": rb.get("error")}
    results, bad = {}, []
    for name, lay, up in _variants(kernel, base):
        vb = build_inputs(base_spec if up is None else {**base_spec, "upstream": up}, diff_args=diff_args(kernel),
                          layout_override={**base_layouts, **lay})
        rv = grad_compare(kernel_fn, ref_fn, vb)
        worst = max((it.get("ratio", 0.0) for it in rv.get("items", []) if isinstance(it.get("ratio"), float)), default=0.0)
        results[name] = {"pass": rv.get("pass"), "failed": rv.get("failed"), "worst_ratio": worst, "error": rv.get("error"),
                         "strides": {k: list(v.stride()) for k, v in vb.tensors.items() if isinstance(v, torch.Tensor)}}
        if not rv.get("pass"):
            bad.append(name)
    return {"status": "fail" if bad else "pass", "violating_variants": bad, "variants": results}


def prop_no_mutation(kernel_fn, ref_fn, spec, kernel):
    meta = ARGS[kernel]
    found, info = {}, {}
    for mode in ("plain", "no_grad"):
        built = build_inputs(spec, diff_args=diff_args(kernel), requires_grad_override=False)
        args = _fresh(built)
        snap = {k: v.detach().clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
        try:
            if mode == "no_grad":
                with torch.no_grad():
                    kernel_fn(**args)
            else:
                kernel_fn(**args)
        except Exception as e:
            return {"status": "error", "reason": f"{type(e).__name__}: {e}"[:500]}
        for k, s in snap.items():
            if not _same(args[k], s):
                ch = (args[k].detach() != s)
                rows = ch.reshape(ch.shape[0], -1).any(1).nonzero().flatten().tolist() if ch.dim() >= 1 else []
                found[f"{k} ({mode}, requires_grad=False)"] = {"elements_changed": int(ch.sum()), "rows_changed": rows[:50]}
    # with gradients enabled: non-differentiable inputs must not change
    built = build_inputs(spec, diff_args=diff_args(kernel))
    args = _fresh(built)
    snap = {k: v.detach().clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    try:
        _run(kernel_fn, args, built.upstream, 0)
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"[:500]}
    for k, s in snap.items():
        if not _same(args[k], s):
            n = int((args[k].detach() != s).sum())
            if args[k].requires_grad:
                info[f"{k} (requires_grad=True)"] = {"elements_changed": n, "note": "write into an input that requires grad"}
            else:
                found[f"{k} (gradients enabled, no gradient for this input)"] = {"elements_changed": n}
    return {"status": "fail" if found else "pass", "violations": found, "info": info}


def prop_compile(kernel_fn, ref_fn, spec, kernel):
    built = build_inputs(spec, diff_args=diff_args(kernel))
    re = grad_compare(kernel_fn, ref_fn, built)
    backend = os.environ.get("KV_COMPILE_BACKEND") or ("inductor" if torch.cuda.is_available() else "aot_eager")
    try:
        torch._dynamo.reset()
        compiled = torch.compile(kernel_fn, backend=backend, dynamic=False)
        rc = grad_compare(compiled, ref_fn, built)
    except Exception as e:
        return {"status": "error", "backend": backend, "reason": f"torch.compile failed: {type(e).__name__}: {e}"[:800]}
    if rc.get("error"):
        return {"status": "error", "backend": backend, "reason": rc["error"][:800]}
    if not re.get("pass"):
        return {"status": "inconclusive", "backend": backend, "reason": "eager already fails the tolerance rule",
                "compiled_pass": rc.get("pass")}
    return {"status": "pass" if rc.get("pass") else "fail", "backend": backend, "compiled_failed": rc.get("failed"),
            "compiled_items": [{k: it.get(k) for k in ("name", "max_abs_err", "allowed", "ratio", "passed")} for it in rc.get("items", [])]}


def prop_determinism(kernel_fn, ref_fn, spec, kernel):
    built = build_inputs(spec, diff_args=diff_args(kernel))
    o1, g1 = _run(kernel_fn, _fresh(built), built.upstream, int(spec.get("seed", 0)))
    o2, g2 = _run(kernel_fn, _fresh(built), built.upstream, int(spec.get("seed", 0)))
    diffs = [f"output{i}" for i, (a, b) in enumerate(zip(o1, o2)) if not _same(a, b)]
    diffs += [f"grad_{k}" for k in g1 if not _same(g1[k], g2.get(k))]
    return {"status": "fail" if diffs else "pass", "nondeterministic": diffs}


_FUNCS = {"zero_upstream": prop_zero_upstream, "layout_invariance": prop_layout_invariance,
          "no_mutation_without_grad": prop_no_mutation, "compile_matches_eager": prop_compile,
          "determinism": prop_determinism}


def check_properties(kernel_fn, ref_fn, spec: dict, kernel: str, which=None) -> dict:
    which = which or PROPERTIES
    out, t0 = {}, time.time()
    for p in which:
        t1 = time.time()
        try:
            out[p] = _FUNCS[p](kernel_fn, ref_fn, spec, kernel)
        except Exception as e:
            out[p] = {"status": "error", "reason": f"{type(e).__name__}: {e}"[:800]}
        out[p]["seconds"] = round(time.time() - t1, 3)
    return {"kernel": kernel, "results": out, "violations": [p for p, r in out.items() if r["status"] == "fail"],
            "seconds": round(time.time() - t0, 3)}
