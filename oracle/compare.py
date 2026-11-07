"""grad_compare: run a kernel in its native dtype and the reference in float64 on the same inputs, and compare every
output and every input gradient under the rule in oracle.tolerances.

    from oracle import grad_compare, build_inputs, load_task, diff_args
    kernel, reference = load_task()
    built = build_inputs(spec, diff_args=diff_args("tvd"))
    report = grad_compare(kernel, reference, built)
    report["pass"], report["items"]       # per output / per gradient: max_abs_err, max_rel_err, allowed, ratio, where
"""

from __future__ import annotations

import time

import torch

from .inputs import Built, clone_like, make_upstream
from .tolerances import allowed_error


def _as_tuple(out):
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)):
        return tuple(o for o in out if isinstance(o, torch.Tensor))
    raise TypeError(f"kernel/reference must return a tensor or tuple of tensors, got {type(out).__name__}")


def _ref_copy(v, dtype):
    if isinstance(v, torch.Tensor):
        if v.dtype.is_floating_point:
            c = v.detach().to(dtype).clone()
            return c.requires_grad_(v.requires_grad)
        return v.detach().clone()
    return v


def reduction_size(args: dict) -> int:
    dims = [d for v in args.values() if isinstance(v, torch.Tensor) for d in v.shape]
    return max(dims) if dims else 1


def compare_tensor(name, kind, T, R, R32, n) -> dict:
    """One comparison under the tolerance rule. T kernel (native dtype), R float64 reference, R32 float32 reference."""
    item = {"name": name, "kind": kind, "dtype": str(T.dtype).replace("torch.", ""), "shape": list(T.shape)}
    if tuple(T.shape) != tuple(R.shape):
        item.update(passed=False, reason=f"shape {list(T.shape)} vs reference {list(R.shape)}")
        return item
    t = T.detach().to("cpu", torch.float64)
    r = R.detach().to("cpu", torch.float64)
    r32 = R32.detach().to("cpu", torch.float64) if R32 is not None else r
    fin_r, fin_t = torch.isfinite(r), torch.isfinite(t)
    if not torch.equal(fin_r, fin_t) or not torch.equal(t[~fin_t], r[~fin_r]):
        bad = (fin_r != fin_t).nonzero()
        item.update(passed=False, reason="non-finite values differ from the reference",
                    n_nonfinite_kernel=int((~fin_t).sum()), n_nonfinite_ref=int((~fin_r).sum()),
                    first_mismatch=bad[0].tolist() if len(bad) else None)
        return item
    diff = (t - r).abs().where(fin_r, torch.zeros_like(r))
    ref_max = float(r.abs().where(fin_r, torch.zeros_like(r)).max()) if r.numel() else 0.0
    err = float(diff.max()) if diff.numel() else 0.0
    err32 = float((r32 - r).abs().where(fin_r & torch.isfinite(r32), torch.zeros_like(r)).max()) if r.numel() else 0.0
    allowed = allowed_error(T.dtype, n, ref_max, err32)
    item.update(max_abs_err=err, ref_max_abs=ref_max, max_rel_err=(err / ref_max if ref_max > 0 else (0.0 if err == 0 else float("inf"))),
                err32=err32, allowed=allowed, ratio=(err / allowed if allowed > 0 else (0.0 if err == 0 else float("inf"))),
                passed=bool(err <= allowed))
    if diff.numel() and err > 0:
        idx = [int(i) for i in torch.unravel_index(diff.argmax(), diff.shape)] if diff.dim() else []
        item.update(where=idx, kernel_at=float(t[tuple(idx)]), ref_at=float(r[tuple(idx)]))
    return item


def grad_compare(kernel_fn, ref_fn, inputs, upstream: dict | None = None, n: int | None = None) -> dict:
    """Compare kernel_fn (native dtype) against ref_fn (float64) on `inputs` (a Built from build_inputs, or a dict of
    call arguments). Gradients are taken for every tensor argument with requires_grad=True, with the spec's upstream
    gradient (default ones). Returns a JSON-serializable report; report['pass'] is the verdict."""
    t0 = time.time()
    if isinstance(inputs, Built):
        args, up, seed = inputs.call_args(), (upstream or inputs.upstream), int(inputs.spec.get("seed", 0))
    else:
        args, up, seed = dict(inputs), (upstream or {"kind": "ones"}), 0
    n = n or reduction_size(args)
    # reference copies FIRST: some kernels write into their inputs
    r64 = {k: _ref_copy(v, torch.float64) for k, v in args.items()}
    r32 = {k: _ref_copy(v, torch.float32) for k, v in args.items()}
    kin = {k: clone_like(v) if isinstance(v, torch.Tensor) else v for k, v in args.items()}
    diff_names = [k for k, v in kin.items() if isinstance(v, torch.Tensor) and v.requires_grad]
    report = {"pass": False, "items": [], "n": n, "diff_args": diff_names}
    try:
        out_k = _as_tuple(kernel_fn(**kin))
    except Exception as e:  # a crash is reported, but it is not a silent-gradient violation
        report.update(error=f"kernel raised {type(e).__name__}: {e}"[:2000], seconds=round(time.time() - t0, 3))
        return report
    try:
        out_r = _as_tuple(ref_fn(**r64))
        with torch.no_grad():
            out_r32 = _as_tuple(ref_fn(**{k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in r32.items()}))
    except Exception as e:
        report.update(error=f"reference raised {type(e).__name__}: {e} (input outside the reference's domain?)"[:2000],
                      seconds=round(time.time() - t0, 3))
        return report
    if len(out_k) != len(out_r):
        report.update(error=f"kernel returned {len(out_k)} tensors, reference {len(out_r)}")
        return report
    items = []
    for i, (T, R, R32) in enumerate(zip(out_k, out_r, out_r32)):
        items.append(compare_tensor(f"output{i}" if len(out_k) > 1 else "output", "output", T, R, R32, n))

    if diff_names:
        req = [i for i, R in enumerate(out_r) if R.requires_grad]
        if any(not out_k[i].requires_grad for i in req):
            items.append({"name": "backward", "kind": "gradient", "passed": False,
                          "reason": "kernel output does not require grad although differentiable inputs do"})
        else:
            gk = [make_upstream(up, out_k[i], out_k[i].dtype, seed) for i in req]
            gr = [make_upstream(up, out_r[i], torch.float64, seed) for i in req]
            grads_k = torch.autograd.grad([out_k[i] for i in req], [kin[a] for a in diff_names], gk, allow_unused=True)
            grads_r = torch.autograd.grad([out_r[i] for i in req], [r64[a] for a in diff_names], gr, allow_unused=True)
            # float32 reference gradients for the data-dependent floor
            r32g = {k: (_ref_copy(v, torch.float32) if isinstance(v, torch.Tensor) else v) for k, v in r32.items()}
            out_r32g = _as_tuple(ref_fn(**r32g))
            g32 = [make_upstream(up, out_r32g[i], torch.float32, seed) for i in req]
            grads_r32 = torch.autograd.grad([out_r32g[i] for i in req], [r32g[a] for a in diff_names], g32, allow_unused=True)
            for a, Gk, Gr, G32 in zip(diff_names, grads_k, grads_r, grads_r32):
                if Gk is None and Gr is None:
                    continue
                if Gk is None or Gr is None:
                    items.append({"name": f"grad_{a}", "kind": "gradient", "passed": False,
                                  "reason": "kernel gave no gradient" if Gk is None else "kernel gave a gradient the reference does not"})
                    continue
                items.append(compare_tensor(f"grad_{a}", "gradient", Gk, Gr, G32, n))
    report["items"] = items
    report["pass"] = all(it.get("passed") for it in items)
    report["failed"] = [it["name"] for it in items if not it.get("passed")]
    report["upstream"] = up
    report["seconds"] = round(time.time() - t0, 3)
    return report
