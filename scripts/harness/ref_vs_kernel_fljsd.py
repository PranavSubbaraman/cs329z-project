"""HARNESS-ONLY: agreement between benchmark/kernels/fljsd/entry.kernel (native dtype) and reference.reference
(float64 copies) on random configurations drawn across the whole documented in-scope domain.

    scripts/harness/in_tree.sh <sha> .venv/bin/python scripts/harness/ref_vs_kernel_fljsd.py [--n 30]

Results for each tree are merged into harness_only/phase1/fljsd_ref_vs_kernel.json under the tree's sha.
The draws validate the reference; they are generic (randn activations, weights randn * s / sqrt(H) with s
log-uniform in [0.5, 5], i.e. logit std 0.5-5) and are not aimed at any particular defect.

Metrics: loss |k - r| <= atol + rtol |r|; each gradient: normwise rel = max|k - r| / max|r| <= tol
(when every row is ignored the reference gradient is exactly zero and the kernel's must be exactly zero too);
teacher .grad must stay None.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KDIR = os.path.join(ROOT, "benchmark", "kernels", "fljsd")
OUT = os.path.join(ROOT, "harness_only", "phase1", "fljsd_ref_vs_kernel.json")

# dtype-appropriate tolerances (bf16: 8 significant bits, unit roundoff 2^-8 = 0.39%; fp16: 11 bits, 0.049%)
TOL = {
    torch.float32: {"loss_rtol": 1e-5, "loss_atol": 1e-6, "grad": 1e-4},
    torch.bfloat16: {"loss_rtol": 1e-2, "loss_atol": 1e-3, "grad": 2e-2},
    torch.float16: {"loss_rtol": 2e-3, "loss_atol": 1e-4, "grad": 5e-3},
}


def _load(name):
    spec = importlib.util.spec_from_file_location(f"fljsd_{name}", os.path.join(KDIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_configs(n, seed):
    rng = random.Random(seed)
    dtypes = [torch.float32, torch.bfloat16, torch.float16]
    cfgs = []
    for i in range(n):
        dt = dtypes[i % 3]
        wide_vocab = i in (5, 16, 27)  # V > 4096: several column blocks per row in the inner kernel
        BT = rng.choice([1, 2, 3, 5, 7, 12, 19, 33, 47, 64, 70])
        Hs = rng.randint(1, 80)
        Ht = Hs if rng.random() < 0.3 else rng.randint(1, 80)
        V = rng.randint(4097, 5300) if wide_vocab else rng.choice([2, 3, rng.randint(4, 64), rng.randint(65, 700)])
        if wide_vocab:
            BT = min(BT, 7)
        r = rng.random()
        beta = 0.0 if r < 0.2 else 1.0 if r < 0.4 else 0.5 if r < 0.55 else round(rng.uniform(0.02, 0.98), 3)
        temperature = 1.0 if rng.random() < 0.4 else round(rng.uniform(0.3, 3.0), 3)
        r = rng.random()
        labels = "none" if r < 0.35 else "partial" if r < 0.85 else "all"
        ignore_index = -100 if rng.random() < 0.6 else rng.randint(0, V - 1)
        accum = None if rng.random() < 0.6 else "float32"
        grad_out = 1.0 if rng.random() < 0.5 else round(rng.choice([-1, 1]) * rng.uniform(0.2, 3.0), 3)
        layout = rng.choice(["contiguous", "contiguous", "row_slices"])
        scale = round(math.exp(rng.uniform(math.log(0.5), math.log(5.0))), 3)
        teacher_requires_grad = rng.random() < 0.3
        cfgs.append(dict(idx=i, dtype=str(dt).replace("torch.", ""), BT=BT, Hs=Hs, Ht=Ht, V=V, jsd_beta=beta,
                         temperature=temperature, labels=labels, ignore_index=ignore_index, accum_dtype=accum,
                         grad_out=grad_out, layout=layout, scale=scale, teacher_requires_grad=teacher_requires_grad,
                         seed=rng.randint(0, 2**31 - 1)))
    return cfgs


def make_tensors(c, device):
    g = torch.Generator().manual_seed(c["seed"])
    BT, Hs, Ht, V, s = c["BT"], c["Hs"], c["Ht"], c["V"], c["scale"]
    xs = torch.randn(BT, Hs, generator=g)
    ws = torch.randn(V, Hs, generator=g) * (s / Hs**0.5)
    xt = torch.randn(BT, Ht, generator=g)
    wt = torch.randn(V, Ht, generator=g) * (s / Ht**0.5)
    lab = None
    if c["labels"] != "none":
        vals = torch.randint(0, V, (BT,), generator=g)
        if c["ignore_index"] >= 0:  # keep "kept" rows distinct from an ignore_index that is a vocab id
            vals[vals == c["ignore_index"]] = (c["ignore_index"] + 1) % V
        if c["labels"] == "all":
            vals[:] = c["ignore_index"]
        else:
            k = max(1, BT // 3)
            idx = torch.randperm(BT, generator=g)[:k]
            vals[idx] = c["ignore_index"]
            if BT > 1 and bool((vals == c["ignore_index"]).all()):
                vals[0] = (c["ignore_index"] + 1) % V if c["ignore_index"] >= 0 else 0
        lab = vals.to(device)
    return [t.to(device) for t in (xs, ws, xt, wt)], lab


def as_layout(t, layout, dtype):
    """Return a tensor equal to t (cast to dtype) in the requested layout (all keep a unit last-dim stride)."""
    t = t.to(dtype)
    if layout == "contiguous":
        return t.clone()
    # row slices of a wider, row-padded buffer: shape (R, C), strides (C + 3, 1); rows interleaved with padding rows
    R, C = t.shape
    buf = torch.full((2 * R, C + 3), float("nan"), dtype=dtype, device=t.device)
    view = buf[::2, :C]
    view.copy_(t)
    return view


def compare(c, kernel, reference, device):
    dt = getattr(torch, c["dtype"])
    (xs, ws, xt, wt), lab = make_tensors(c, device)
    accum = torch.float32 if c["accum_dtype"] == "float32" else None
    kw = dict(shift_labels=lab, jsd_beta=c["jsd_beta"], ignore_index=c["ignore_index"], temperature=c["temperature"],
              accum_dtype=accum)
    # kernel: native dtype, student tensors as leaves in the requested layout
    xs_k = as_layout(xs, c["layout"], dt).requires_grad_(True)
    ws_k = as_layout(ws, c["layout"], dt).requires_grad_(True)
    xt_k = as_layout(xt, c["layout"], dt).requires_grad_(c["teacher_requires_grad"])
    wt_k = as_layout(wt, c["layout"], dt).requires_grad_(c["teacher_requires_grad"])
    t0 = time.time()
    loss_k = kernel(xs_k, ws_k, xt_k, wt_k, **kw)
    (loss_k * c["grad_out"]).backward() if c["grad_out"] != 1.0 else loss_k.backward()
    secs = time.time() - t0
    # reference: float64 copies of exactly the same (already rounded) values
    xs_r = xs_k.detach().double().requires_grad_(True)
    ws_r = ws_k.detach().double().requires_grad_(True)
    xt_r = xt_k.detach().double().requires_grad_(c["teacher_requires_grad"])
    wt_r = wt_k.detach().double().requires_grad_(c["teacher_requires_grad"])
    loss_r = reference(xs_r, ws_r, xt_r, wt_r, **kw)
    (loss_r * c["grad_out"]).backward()

    tol = TOL[dt]
    res = dict(c)
    res["seconds"] = round(secs, 3)
    res["loss_kernel_dtype"] = str(loss_k.dtype).replace("torch.", "")
    lk, lr = loss_k.item(), loss_r.item()
    res["loss_abs_err"] = abs(lk - lr)
    res["loss_ok"] = bool(math.isfinite(lk) and abs(lk - lr) <= tol["loss_atol"] + tol["loss_rtol"] * abs(lr))
    ok = res["loss_ok"]
    for name, k, r in (("grad_input", xs_k.grad, xs_r.grad), ("grad_weight", ws_k.grad, ws_r.grad)):
        if k is None:
            res[f"{name}_rel"] = None
            res[f"{name}_ok"] = False
            ok = False
            continue
        res[f"{name}_dtype_ok"] = bool(k.dtype == dt and k.shape == r.shape)
        rmax = r.abs().max().item()
        if rmax == 0.0:
            res[f"{name}_rel"] = 0.0 if bool((k == 0).all()) else float("inf")
        else:
            res[f"{name}_rel"] = ((k.double() - r).abs().max().item()) / rmax
        res[f"{name}_ok"] = bool(res[f"{name}_dtype_ok"] and res[f"{name}_rel"] <= tol["grad"])
        ok = ok and res[f"{name}_ok"]
    res["teacher_grad_none"] = bool(xt_k.grad is None and wt_k.grad is None)
    ok = ok and res["teacher_grad_none"]
    res["ok"] = bool(ok)
    return res


def conditioning_probe(kernel, reference, device):
    """Supplementary, NOT part of all_ok: float32 only, interior jsd_beta, large logit spreads. When student and
    teacher are both peaked on different tokens the interior-beta JSD saturates and its true gradient vanishes, so
    the logits-side gradient dX - q * sum(dX) cancels catastrophically in any float32 evaluation. Reports the kernel's
    normwise error next to the same-precision baseline (reference.py itself evaluated on float32 copies)."""
    rows = []
    g = torch.Generator().manual_seed(7)
    for beta in (0.5, 0.25):
        for std in (3.0, 10.0, 30.0):
            BT, H, V = 8, 64, 5000
            xs = torch.randn(BT, H, generator=g)
            ws = torch.randn(V, H, generator=g) * (std / H**0.5)
            xt = torch.randn(BT, H, generator=g)
            wt = torch.randn(V, H, generator=g) * (std / H**0.5)
            out = {}
            for name, fn, dt in (("ref64", reference, torch.float64), ("kernel32", kernel, torch.float32),
                                 ("ref32", reference, torch.float32)):
                a = xs.to(device, dt).clone().requires_grad_(True)
                b = ws.to(device, dt).clone().requires_grad_(True)
                fn(a, b, xt.to(device, dt).clone(), wt.to(device, dt).clone(), jsd_beta=beta).backward()
                out[name] = (a.grad.double(), b.grad.double())
            row = {"jsd_beta": beta, "logit_std": std, "BT": BT, "H": H, "V": V,
                   "max_abs_ref_grad_input": out["ref64"][0].abs().max().item()}
            for which, i in (("grad_input", 0), ("grad_weight", 1)):
                r = out["ref64"][i]
                for name in ("kernel32", "ref32"):
                    row[f"{which}_{name}_rel"] = ((out[name][i] - r).abs().max() / r.abs().max()).item()
            row["kernel_within_4x_of_float32_baseline"] = bool(all(
                row[f"{w}_kernel32_rel"] <= 4 * row[f"{w}_ref32_rel"] + 1e-4 for w in ("grad_input", "grad_weight")))
            rows.append(row)
            print(f"[cond] beta={beta} std={std}: max|ref gi|={row['max_abs_ref_grad_input']:.2e} "
                  f"kernel32 gi={row['grad_input_kernel32_rel']:.2e} ref32 gi={row['grad_input_ref32_rel']:.2e} "
                  f"kernel32 gw={row['grad_weight_kernel32_rel']:.2e} ref32 gw={row['grad_weight_ref32_rel']:.2e}",
                  file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, KDIR)
    kernel = _load("entry").kernel
    reference = _load("reference").reference
    tree = os.path.basename(os.environ.get("LIGER_TREE", "unknown"))
    backend = torch.cuda.get_device_name(0) if device == "cuda" else (
        "interpreter-cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cpu")
    rows = []
    t0 = time.time()
    for c in draw_configs(args.n, args.seed):
        try:
            r = compare(c, kernel, reference, device)
        except Exception as e:  # noqa: BLE001
            r = dict(c, ok=False, error=f"{type(e).__name__}: {e}")
        rows.append(r)
        print(f"[{r['idx']:2d}] {r['dtype']:8s} BT={r['BT']:3d} Hs={r['Hs']:2d} Ht={r['Ht']:2d} V={r['V']:4d} "
              f"beta={r['jsd_beta']} T={r['temperature']} lab={r['labels']:7s} acc={r['accum_dtype']} "
              f"g={r['grad_out']} {r['layout']:10s} s={r['scale']}: "
              f"loss_err={r.get('loss_abs_err', float('nan')):.2e} gi={r.get('grad_input_rel', float('nan'))} "
              f"gw={r.get('grad_weight_rel', float('nan'))} -> {'OK' if r['ok'] else 'MISMATCH'} {r.get('error', '')}",
              file=sys.stderr, flush=True)

    def worst(dt, key):
        vals = [r[key] for r in rows if r["dtype"] == dt and r.get(key) is not None]
        return max(vals) if vals else None

    summary = {
        "tree": tree,
        "backend": backend,
        "n_configs": len(rows),
        "n_ok": sum(r["ok"] for r in rows),
        "all_ok": all(r["ok"] for r in rows),
        "mismatches": [r["idx"] for r in rows if not r["ok"]],
        "tolerances": {str(k).replace("torch.", ""): v for k, v in TOL.items()},
        "worst": {dt: {"grad_input_rel": worst(dt, "grad_input_rel"), "grad_weight_rel": worst(dt, "grad_weight_rel"),
                       "loss_abs_err": worst(dt, "loss_abs_err")} for dt in ("float32", "bfloat16", "float16")},
        "options_covered": {
            "dtype": sorted({r["dtype"] for r in rows}),
            "jsd_beta": sorted({("0" if r["jsd_beta"] == 0 else "1" if r["jsd_beta"] == 1 else "interior")
                                for r in rows}),
            "labels": sorted({r["labels"] for r in rows}),
            "accum_dtype": sorted({str(r["accum_dtype"]) for r in rows}),
            "temperature_ne_1": any(r["temperature"] != 1.0 for r in rows),
            "grad_out_ne_1": any(r["grad_out"] != 1.0 for r in rows),
            "layouts": sorted({r["layout"] for r in rows}),
            "Hs_ne_Ht": any(r["Hs"] != r["Ht"] for r in rows),
            "V_gt_4096": any(r["V"] > 4096 for r in rows),
            "ignore_index_is_vocab_id": any(r["ignore_index"] >= 0 and r["labels"] != "none" for r in rows),
        },
        "seconds": round(time.time() - t0, 1),
        "torch": torch.__version__,
    }
    allres = {}
    if os.path.exists(args.out):
        try:
            allres = json.load(open(args.out))
        except json.JSONDecodeError:
            allres = {}
    cond = conditioning_probe(kernel, reference, device)
    summary["conditioning_probe_float32"] = {
        "note": "supplementary, not part of all_ok; see conditioning_probe() docstring",
        "kernel_within_4x_of_float32_baseline_everywhere": all(r["kernel_within_4x_of_float32_baseline"] for r in cond),
        "worst_kernel32_rel": max(max(r["grad_input_kernel32_rel"], r["grad_weight_kernel32_rel"]) for r in cond),
        "worst_ref32_rel": max(max(r["grad_input_ref32_rel"], r["grad_weight_ref32_rel"]) for r in cond),
    }
    allres[tree] = {"summary": summary, "rows": rows, "conditioning_probe_float32": cond}
    with open(args.out, "w") as f:
        json.dump(allres, f, indent=1)
    print(json.dumps(summary))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
