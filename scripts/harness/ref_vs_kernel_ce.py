"""HARNESS-ONLY: kernel-vs-reference agreement check for benchmark/kernels/ce.

Run on a tree:  scripts/harness/in_tree.sh <sha-prefix> .venv/bin/python scripts/harness/ref_vs_kernel_ce.py

Part 1 (pass/fail): 33 random configurations, plus a 12-config boundary grid (BT in {1, 2}, V in {2, 3}, each
dtype and reduction), over the whole documented in-scope domain of task.md: each input
dtype (float32, float16) x each reduction, non-power-of-2 sizes (plus one power-of-2 draw), one vocabulary larger
than the kernel's 32768-wide block, ignore_index = -100 / -1 / an in-range class id, no / some / all rows ignored,
label smoothing (incl. 1.0), z-loss scale, upstream gradient 1 or random (scalar or per row), contiguous or
row-strided input (column slice of a wider tensor), contiguous or strided target, forward-only calls. The kernel
runs in the input's native dtype; the reference runs on float64 copies of the same values with the same (rounded)
upstream gradient. Outputs and input gradients are compared with dtype-appropriate tolerances. These draws
validate the reference; they are not aimed at any bug.

Also recorded per config, for information only (never part of pass/fail): whether the call left the caller's
`input` tensor modified (by design when it requires grad; see harness_only/phase1/ce_notes.md).

Part 2 (informational, not pass/fail): probes of options that are OUT of the documented scope, kept here as
evidence for why they are out of scope: input with stride(-1) != 1, bfloat16 input with an upstream gradient
other than 1, class weights, softcap.

Prints one summary JSON line and merges the full record into harness_only/phase1/ce_ref_vs_kernel.json under
runs["<tree>@<backend>"].
"""

import importlib.util
import json
import os
import pathlib
import random
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
KDIR = ROOT / "benchmark" / "kernels" / "ce"
OUT = ROOT / "harness_only" / "phase1" / "ce_ref_vs_kernel.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep benchmark/kernels/ce free of __pycache__
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dwb
    return mod


entry = _load("ce_entry", KDIR / "entry.py")
ref_mod = _load("ce_reference", KDIR / "reference.py")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = {"float32": torch.float32, "float16": torch.float16}
REDUCTIONS = ("mean", "sum", "none")
# (loss rtol, grad rtol, grad atol as a fraction of max|ref grad|); loss atol = loss rtol * max|ref loss|
TOL = {
    "float32": (2e-5, 1e-4, 1e-5),
    "float16": (4e-3, 4e-3, 1e-3),
}


def backend():
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return f"interpreter-{DEVICE}"
    return torch.cuda.get_device_name() if DEVICE == "cuda" else DEVICE


# ------------------------------------------------------------------------------------------------
# Part 1: in-scope random draws
# ------------------------------------------------------------------------------------------------
def draw_configs(seed=20261106, n_extra=27):
    rng = random.Random(seed)
    cfgs = []
    base = [(d, r) for d in DTYPES for r in REDUCTIONS]  # every (dtype, reduction) pair at least once
    pairs = base + [(rng.choice(list(DTYPES)), rng.choice(REDUCTIONS)) for _ in range(n_extra)]
    for i, (dtype, reduction) in enumerate(pairs):
        V = rng.choice([rng.randint(2, 40), rng.randint(41, 700), rng.randint(701, 5000)])
        if V & (V - 1) == 0:
            V += 1  # non-power-of-2 vocabulary sizes (one power-of-2 draw is added explicitly below)
        cfgs.append(
            {
                "id": i,
                "dtype": dtype,
                "reduction": reduction,
                "BT": rng.randint(1, 70),
                "V": V,
                "scale": rng.choice([0.5, 1.0, 1.0, 4.0, 16.0]),
                "ignore_index": rng.choice([-100, -100, -100, -1, "in_range"]),
                "ignore_frac": rng.choice([0.0, 0.1, 0.3, 0.5, 0.5]),
                "label_smoothing": rng.choice([0.0, 0.0, round(rng.uniform(0.01, 0.99), 3), 1.0]),
                "lse_square_scale": rng.choice([0.0, 0.0, 1e-4, 1e-3, 1e-2]),
                "requires_grad": rng.random() > 0.2,
                "upstream": rng.choice(["one", "random"]),
                "layout": rng.choice(["contiguous", "contiguous", "contiguous", "row_slice"]),
                "target_layout": rng.choice(["contiguous", "contiguous", "contiguous", "strided"]),
            }
        )
    # explicit coverage that random draws might miss
    cfgs[1].update(V=32768 + 1237, BT=3, label_smoothing=0.2, lse_square_scale=1e-3, requires_grad=True)  # 2 blocks
    cfgs[4].update(ignore_frac=1.0, requires_grad=True, upstream="random")  # every row ignored
    cfgs[6].update(V=256, BT=64, ignore_frac=0.0)  # power-of-2 sizes
    cfgs[7].update(ignore_frac=1.0, requires_grad=True)  # every row ignored
    # deterministic boundary grid of the documented domain: BT in {1, 2}, V in {2, 3}, each dtype and reduction
    for dtype in DTYPES:
        for reduction in REDUCTIONS:
            for BT, V, frac in ((1, 2, 0.0), (2, 3, 0.5)):
                cfgs.append(
                    {
                        "id": -1,
                        "dtype": dtype,
                        "reduction": reduction,
                        "BT": BT,
                        "V": V,
                        "scale": 1.0,
                        "ignore_index": -100,
                        "ignore_frac": frac,
                        "label_smoothing": 0.1,
                        "lse_square_scale": 1e-3,
                        "requires_grad": True,
                        "upstream": "random",
                        "layout": "contiguous",
                        "target_layout": "contiguous",
                    }
                )
    for k, c in enumerate(cfgs):
        c["id"] = k
    return cfgs


def build_inputs(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    BT, V = cfg["BT"], cfg["V"]
    vals = (torch.randn(BT, V, generator=g, dtype=torch.float64) * cfg["scale"]).to(DTYPES[cfg["dtype"]])
    ii = cfg["ignore_index"]
    if ii == "in_range":
        ii = int(torch.randint(0, V, (1,), generator=g))
    t = torch.randint(0, V, (BT,), generator=g)
    n_ign = int(round(cfg["ignore_frac"] * BT))
    if 0 < cfg["ignore_frac"] < 1 and n_ign == 0 and BT > 1:
        n_ign = 1
    if n_ign:
        t[torch.randperm(BT, generator=g)[:n_ign]] = ii
    rnd = cfg["upstream"] == "random"
    if cfg["reduction"] == "none":
        up = torch.randn(BT, generator=g, dtype=torch.float64) if rnd else torch.ones(BT, dtype=torch.float64)
    else:
        up = torch.rand((), generator=g, dtype=torch.float64) * 1.5 + 0.5 if rnd else torch.tensor(1.0, dtype=torch.float64)
    return vals, t, ii, up


def kernel_input(vals, layout):
    """A leaf tensor on DEVICE with the same values as `vals` in the requested memory layout."""
    vals = vals.to(DEVICE)
    if layout == "contiguous":
        x = vals.clone()
    elif layout == "row_slice":  # rows strided (stride(-2) = V + 7), last dim unit-stride
        big = torch.zeros(vals.shape[0], vals.shape[1] + 7, dtype=vals.dtype, device=DEVICE)
        big[:, 3 : 3 + vals.shape[1]] = vals
        x = big[:, 3 : 3 + vals.shape[1]]
    elif layout == "last_dim_strided":  # stride(-1) != 1 (OUT of scope; used by a probe only)
        x = vals.t().contiguous().t()
    else:
        raise ValueError(layout)
    assert torch.equal(x, vals)
    return x


def target_tensor(t, layout):
    t = t.to(DEVICE)
    if layout == "strided":
        big = torch.zeros(2 * t.shape[0], dtype=t.dtype, device=DEVICE)
        big[::2] = t
        return big[::2]
    return t.clone()


def compare(k, r, rtol, atol_frac):
    k64, r64 = k.detach().double().cpu(), r.detach().double().cpu()
    scale = float(r64.abs().max()) if r64.numel() else 0.0
    atol = atol_frac * scale if scale > 0 else 1e-30
    err = (k64 - r64).abs()
    ok = bool(torch.all(err <= atol + rtol * r64.abs()))
    return ok, {
        "max_abs_err": float(err.max()) if err.numel() else 0.0,
        "max_err_over_max_ref": float(err.max() / scale) if scale > 0 and err.numel() else 0.0,
        "ref_max_abs": scale,
        "finite": bool(torch.isfinite(k64).all()),
    }


def kernel_and_reference(vals, t, up, layout, target_layout, requires_grad, kw):
    xk = kernel_input(vals, layout)
    tk = target_tensor(t, target_layout)
    if requires_grad:
        xk.requires_grad_(True)
    snapshot = xk.detach().clone()
    out_k = entry.kernel(xk, tk, **kw)
    rows_changed = int((xk.detach() != snapshot).any(dim=1).sum())
    grad_k = None
    if requires_grad:
        out_k.backward(up.to(out_k.dtype).to(DEVICE))  # scalar for mean/sum, one value per row for none
        grad_k = xk.grad
    xr = vals.double().clone().requires_grad_(True)
    out_r = ref_mod.reference(xr, t, **kw)
    (grad_r,) = torch.autograd.grad(out_r, xr, grad_outputs=up.to(out_k.dtype).double())
    return out_k, grad_k, out_r, grad_r, rows_changed


def run_one(cfg):
    vals, t, ii, up = build_inputs(cfg, seed=1000 + cfg["id"])
    lrtol, grtol, gatol = TOL[cfg["dtype"]]
    kw = dict(
        ignore_index=ii,
        reduction=cfg["reduction"],
        label_smoothing=cfg["label_smoothing"],
        lse_square_scale=cfg["lse_square_scale"],
    )
    out_k, grad_k, out_r, grad_r, rows_changed = kernel_and_reference(
        vals, t, up, cfg["layout"], cfg["target_layout"], cfg["requires_grad"], kw
    )
    ok_l, m_l = compare(out_k, out_r, lrtol, lrtol)
    res = {
        "cfg": dict(cfg, ignore_index_value=ii, n_ignored=int((t == ii).sum())),
        "loss": m_l,
        "loss_ok": ok_l,
        "shape_ok": tuple(out_k.shape) == tuple(out_r.shape),
        "dtype_ok": out_k.dtype == DTYPES[cfg["dtype"]],
    }
    ok_g = True
    if grad_k is not None:
        ok_g, m_g = compare(grad_k, grad_r, grtol, gatol)
        res.update(grad=m_g, grad_ok=ok_g)
    else:
        res.update(grad=None, grad_ok=None)
    res["input_rows_modified_by_call"] = rows_changed  # informational only
    res["pass"] = bool(ok_l and ok_g and res["shape_ok"] and res["dtype_ok"])
    return res


# ------------------------------------------------------------------------------------------------
# Part 2: out-of-scope probes (informational)
# ------------------------------------------------------------------------------------------------
def probe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def probes():
    g = torch.Generator().manual_seed(7)
    vals32 = torch.randn(6, 50, generator=g, dtype=torch.float64).float()
    t = torch.randint(0, 50, (6,), generator=g)
    t[2] = -100
    out = {}

    def last_dim_strided():
        res = {}
        for dt in ("float32", "float16"):
            vals = vals32.to(DTYPES[dt])
            kw = dict(ignore_index=-100, reduction="mean", label_smoothing=0.0, lse_square_scale=0.0)
            up = torch.tensor(1.0, dtype=torch.float64)
            out_k, grad_k, out_r, grad_r, _ = kernel_and_reference(vals, t, up, "last_dim_strided", "contiguous", True, kw)
            logits_zeroed = vals.double().clone()
            logits_zeroed[t == -100] = 0
            res[dt] = {
                "loss_ok": compare(out_k, out_r, TOL[dt][0], TOL[dt][0])[0],
                "grad_vs_ref": compare(grad_k, grad_r, TOL[dt][1], TOL[dt][2])[1],
                "grad_equals_input_logits_with_ignored_rows_zeroed": bool(
                    torch.allclose(grad_k.double().cpu(), logits_zeroed, atol=0, rtol=0)
                ),
            }
        return res

    def bf16_upstream():
        res = {}
        vals = vals32.to(torch.bfloat16)
        for name, upv in (("upstream_1.0", 1.0), ("upstream_1.75", 1.75)):
            for red in ("mean", "sum"):
                kw = dict(ignore_index=-100, reduction=red, label_smoothing=0.0, lse_square_scale=0.0)
                up = torch.tensor(upv, dtype=torch.float64)
                out_k, grad_k, out_r, grad_r, _ = kernel_and_reference(vals, t, up, "contiguous", "contiguous", True, kw)
                res[f"{red}/{name}"] = {
                    "loss": compare(out_k, out_r, 2e-2, 2e-2)[1],
                    "grad": compare(grad_k, grad_r, 2e-2, 4e-3)[1],
                }
        # the interpreter primitive underneath: bf16 * bf16 inside a Triton kernel
        import triton
        import triton.language as tl

        @triton.jit
        def _mul(X_ptr, G_ptr, N, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            gval = tl.load(G_ptr)
            x = tl.load(X_ptr + offs, mask=offs < N)
            tl.store(X_ptr + offs, x * gval, mask=offs < N)

        y = torch.full((8,), 0.5, dtype=torch.bfloat16, device=DEVICE)
        _mul[(1,)](y, torch.tensor(1.75, dtype=torch.bfloat16, device=DEVICE), 8, BLOCK=8)
        res["triton_bf16_times_bf16_0.5x1.75"] = float(y[0])
        return res

    def liger_direct(**extra):
        def f():
            from liger_kernel.transformers.functional import liger_cross_entropy

            x = vals32.clone().to(DEVICE).requires_grad_(True)
            loss = liger_cross_entropy(input=x, target=t.to(DEVICE), **extra)
            loss.backward()
            return {"ran": True, "loss": float(loss)}

        return f

    out["last_dim_strided_input (stride(-1) != 1)"] = probe(last_dim_strided)
    out["bfloat16_input"] = probe(bf16_upstream)
    out["class_weight"] = probe(liger_direct(weight=torch.rand(50, generator=g).to(DEVICE) + 0.1))
    out["softcap"] = probe(liger_direct(softcap=30.0))
    return out


def main():
    t0 = time.time()
    results = []
    for cfg in draw_configs():
        try:
            results.append(run_one(cfg))
        except Exception as exc:  # noqa: BLE001
            results.append({"cfg": cfg, "pass": False, "error": f"{type(exc).__name__}: {exc}"})
    fails = [r for r in results if not r["pass"]]
    ok = [r for r in results if "error" not in r]
    fwd_only = [r for r in ok if not r["cfg"]["requires_grad"]]
    side = {
        "forward_only_configs": len(fwd_only),
        "forward_only_configs_with_ignored_rows": sum(1 for r in fwd_only if r["cfg"]["n_ignored"] > 0),
        "forward_only_configs_with_input_modified": sum(1 for r in fwd_only if r["input_rows_modified_by_call"] > 0),
        "forward_only_[id, rows_modified, rows_ignored]": [
            [r["cfg"]["id"], r["input_rows_modified_by_call"], r["cfg"]["n_ignored"]] for r in fwd_only
        ],
        "requires_grad_configs_with_input_modified_(by_design)": sum(
            1 for r in ok if r["cfg"]["requires_grad"] and r["input_rows_modified_by_call"] > 0
        ),
        "requires_grad_configs": sum(1 for r in ok if r["cfg"]["requires_grad"]),
    }
    tree = os.path.basename(os.environ.get("LIGER_TREE", "").rstrip("/")) or "unknown"
    summary = {
        "kernel": "ce",
        "tree": tree,
        "backend": backend(),
        "torch": torch.__version__,
        "n_configs": len(results),
        "n_pass": len(results) - len(fails),
        "n_fail": len(fails),
        "failed_ids": [r["cfg"]["id"] for r in fails],
        "coverage": {
            "dtype_x_reduction": sorted({f'{r["cfg"]["dtype"]}/{r["cfg"]["reduction"]}' for r in results}),
            "label_smoothing>0": sum(1 for r in results if r["cfg"]["label_smoothing"] > 0),
            "label_smoothing==1": sum(1 for r in results if r["cfg"]["label_smoothing"] == 1.0),
            "lse_square_scale>0": sum(1 for r in results if r["cfg"]["lse_square_scale"] > 0),
            "forward_only": sum(1 for r in results if not r["cfg"]["requires_grad"]),
            "upstream_random": sum(1 for r in results if r["cfg"]["upstream"] == "random"),
            "layouts": {lay: sum(1 for r in results if r["cfg"]["layout"] == lay) for lay in ("contiguous", "row_slice")},
            "strided_target": sum(1 for r in results if r["cfg"]["target_layout"] == "strided"),
            "ignore_index_kinds": sorted({str(r["cfg"]["ignore_index"]) for r in results}),
            "no_ignored_rows": sum(1 for r in results if r["cfg"]["ignore_frac"] == 0.0),
            "all_rows_ignored": sum(1 for r in results if r["cfg"]["ignore_frac"] == 1.0),
            "power_of_2_V": sum(1 for r in results if r["cfg"]["V"] & (r["cfg"]["V"] - 1) == 0),
            "max_V": max(r["cfg"]["V"] for r in results),
            "BT_range": [min(r["cfg"]["BT"] for r in results), max(r["cfg"]["BT"] for r in results)],
        },
        "tolerances": {k: {"loss_rtol": v[0], "grad_rtol": v[1], "grad_atol_frac_of_max": v[2]} for k, v in TOL.items()},
        "side_effects_informational": side,
        "out_of_scope_probes_informational": probes(),
        "seconds": round(time.time() - t0, 2),
        "results": results,
    }
    data = {}
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            data = {}
    data.setdefault("kernel", "ce")
    data.setdefault("runs", {})[f"{tree}@{summary['backend']}"] = summary
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
