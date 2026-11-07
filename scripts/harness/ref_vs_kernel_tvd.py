"""HARNESS-ONLY: random-configuration agreement check for kernel "tvd".

Compares benchmark/kernels/tvd/entry.py::kernel (native dtype, on whichever Liger tree is on PYTHONPATH) with
benchmark/kernels/tvd/reference.py::reference evaluated in float64 (every floating input cast to float64), on
outputs and on the gradient w.r.t. p, over ~30 random configurations spanning the in-scope domain of task.md:
all four reductions x every allowed (p, q) dtype pair x shift_labels None / given (random ignored fraction, several
ignore_index values), non-power-of-2 BT and V (V also above the 16384 block size), unnormalized and normalized
inputs, contiguous and strided views, scalar upstream gradients (implicit 1.0 or random) and (BT, V) upstream
gradients for "none", and q with or without requires_grad.

These draws validate the reference. They are NOT tuned toward any bug: values are continuous random draws, and
the count of exact p == q elements that occurred by chance is only recorded as a diagnostic.

Agreement criterion, elementwise: |k - r| <= atol + rtol * |r|, with atol = 1e-6 * max|r| (a norm-relative floor;
0 when r is identically 0). rtol(loss) = 1e-5 for every in-scope dtype pair (the kernel computes in float32).
rtol(grad) = 1e-5 when p is float32 and 2e-2 (~2.5 bf16 ulps) when p is bfloat16: the kernel stores the gradient in
bf16 (the Triton CPU interpreter truncates on that store, < 1 ulp = 2^-7), and a non-unit scalar upstream adds two
more torch bf16 roundings, so the worst case is ~2^-6 = 1.6%.
Contract checks: output shape, loss dtype float32, p.grad dtype == p.dtype, q.grad stays None, p and q unmodified.

Run:   scripts/harness/in_tree.sh <sha-prefix> .venv/bin/python scripts/harness/ref_vs_kernel_tvd.py [--n 30] [--seed 0]
Output: one summary JSON line on stdout. harness_only/phase1/tvd_ref_vs_kernel.json is updated, keyed by commit.
Exit 0 if every configuration agrees, 1 otherwise.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "kernels" / "tvd"))
from entry import kernel  # noqa: E402
from reference import reference  # noqa: E402

OUT_JSON = ROOT / "harness_only" / "phase1" / "tvd_ref_vs_kernel.json"
REDUCTIONS = ["none", "sum", "mean", "batchmean"]
DTYPE_PAIRS = [("float32", "float32"), ("bfloat16", "float32"), ("float32", "bfloat16")]
DT = {"float32": torch.float32, "bfloat16": torch.bfloat16}
SIZES_V = [1, 2, 3, 7, 17, 100, 255, 513, 1000, 1271, 4097, 16385, 20000, 33001]
SIZES_BT = [1, 2, 3, 5, 7, 13, 31, 50, 97, 130]
LAYOUTS = ["contiguous", "transposed", "col_slice", "row_stride"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_view(BT, V, layout, gen, kind, dtype):
    """Return (base, view_fn) where view_fn(base) is a (BT, V) tensor with the requested layout."""
    def draw(*shape):
        if kind == "probs":
            return torch.randn(*shape, generator=gen).mul(random_scale(gen)).softmax(-1)
        if kind == "uniform":
            return torch.rand(*shape, generator=gen)
        return torch.randn(*shape, generator=gen) * random_scale(gen)  # unnormalized

    if layout == "contiguous":
        base, fn = draw(BT, V), (lambda t: t)
    elif layout == "transposed":
        base = draw(BT, V).t().contiguous()  # (V, BT) storage
        fn = lambda t: t.t()  # noqa: E731
    elif layout == "col_slice":
        base = draw(BT, V + 3)
        fn = lambda t: t[:, :V]  # noqa: E731
    elif layout == "row_stride":
        base = draw(2 * BT, V)
        fn = lambda t: t[::2]  # noqa: E731
    else:
        raise AssertionError(layout)
    return base.to(dtype).to(DEVICE), fn


def random_scale(gen):
    return [0.1, 1.0, 5.0][int(torch.randint(0, 3, (1,), generator=gen))]


def ratio(k, r, rtol):
    """max over elements of |k - r| / (atol + rtol*|r|), atol = 1e-6 * max|r|; <= 1 means agreement."""
    k = k.detach().double().cpu()
    r = r.detach().double().cpu()
    err = (k - r).abs()
    tol = 1e-6 * float(r.abs().max()) + rtol * r.abs()
    zero_tol = tol == 0
    out = torch.where(zero_tol, torch.where(err == 0, 0.0, math.inf), err / torch.where(zero_tol, 1.0, tol))
    return float(out.max()), float(err.max())


def strict(obj):
    """Make obj strict-JSON safe: inf/nan -> strings (an infinite ratio = reference exactly 0, kernel nonzero)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: strict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strict(v) for v in obj]
    return obj


def sample_configs(n, seed):
    rng = random.Random(seed)
    cfgs = []
    for i in range(n):
        red = REDUCTIONS[i % 4] if i < 24 else rng.choice(REDUCTIONS)
        pdt, qdt = DTYPE_PAIRS[(i // 4) % 3] if i < 24 else rng.choice(DTYPE_PAIRS)
        with_labels = ((i // 12) % 2 == 1) if i < 24 else rng.random() < 0.5
        V = rng.choice(SIZES_V)
        BT = rng.choice(SIZES_BT)
        if V > 10000:
            BT = min(BT, rng.choice([1, 2, 3, 5]))
        cfgs.append({
            "i": i,
            "BT": BT,
            "V": V,
            "reduction": red,
            "p_dtype": pdt,
            "q_dtype": qdt,
            "labels": with_labels,
            "ignore_index": rng.choice([-100, 0, 1, 5]) if with_labels else -100,
            "ignore_frac": rng.choice([0.0, 0.1, 0.3, 0.5, 0.7]) if with_labels else 0.0,
            "p_kind": rng.choice(["unnormalized", "probs", "uniform"]),
            "q_kind": rng.choice(["probs", "probs", "unnormalized"]),
            "p_layout": rng.choice(LAYOUTS),
            "q_layout": rng.choice(LAYOUTS),
            "upstream": "tensor" if red == "none" else rng.choice(["implicit_1.0", "random_scalar"]),
            "q_requires_grad": rng.random() < 0.3,
            "seed": seed * 1000 + i,
        })
    return cfgs


def run_one(cfg):
    gen = torch.Generator().manual_seed(cfg["seed"])
    BT, V, red = cfg["BT"], cfg["V"], cfg["reduction"]
    pdt, qdt = DT[cfg["p_dtype"]], DT[cfg["q_dtype"]]
    base_p, fp = make_view(BT, V, cfg["p_layout"], gen, cfg["p_kind"], pdt)
    base_q, fq = make_view(BT, V, cfg["q_layout"], gen, cfg["q_kind"], qdt)
    labels = None
    ii = cfg["ignore_index"]
    if cfg["labels"]:
        while True:
            labels = torch.randint(-1, max(V, 2), (BT,), generator=gen, dtype=torch.long)
            labels[torch.rand(BT, generator=gen) < cfg["ignore_frac"]] = ii
            if red in ("none", "sum") or bool((labels != ii).any()):
                break
        labels = labels.to(DEVICE)
    if red == "none":
        up = torch.randn(BT, V, generator=gen).to(DEVICE)
    elif cfg["upstream"] == "random_scalar":
        up = torch.tensor(float(torch.empty(1).uniform_(-3, 3, generator=gen)), device=DEVICE)
    else:
        up = None

    # kernel, native dtype
    bp = base_p.clone().requires_grad_(True)
    bq = base_q.clone().requires_grad_(cfg["q_requires_grad"])
    p, q = fp(bp), fq(bq)
    t0 = time.time()
    out = kernel(p, q, shift_labels=labels, reduction=red, ignore_index=ii)
    out.backward(up) if up is not None else out.backward()
    secs = time.time() - t0
    gk = fp(bp.grad)

    # reference, float64
    bp64 = base_p.detach().double().requires_grad_(True)
    bq64 = base_q.detach().double().requires_grad_(cfg["q_requires_grad"])
    ref = reference(fp(bp64), fq(bq64), shift_labels=labels, reduction=red, ignore_index=ii)
    ref.backward(up.double()) if up is not None else ref.backward()
    gr = fp(bp64.grad)

    rtol_loss = 1e-5
    rtol_grad = 1e-5 if cfg["p_dtype"] == "float32" else 2e-2
    loss_ratio, loss_abs = ratio(out, ref, rtol_loss)
    grad_ratio, grad_abs = ratio(gk, gr, rtol_grad)
    contract = {
        "shape_ok": tuple(out.shape) == tuple(ref.shape),
        "loss_dtype_float32": out.dtype == torch.float32,
        "grad_dtype_is_p_dtype": gk.dtype == pdt,
        "q_grad_none": bq.grad is None and bq64.grad is None,
        "inputs_unmodified": bool(torch.equal(bp.detach(), base_p) and torch.equal(bq.detach(), base_q)),
    }
    agree = loss_ratio <= 1 and grad_ratio <= 1 and all(contract.values())
    n_kept = BT if labels is None else int((labels != ii).sum())
    return {
        **cfg,
        "n_kept_rows": n_kept,
        "n_exact_ties": int((p.detach().double() == q.detach().double()).sum()),
        "p_contiguous": p.is_contiguous(),
        "q_contiguous": q.is_contiguous(),
        "loss_ratio": loss_ratio,
        "loss_max_abs_err": loss_abs,
        "grad_ratio": grad_ratio,
        "grad_max_abs_err": grad_abs,
        "grad_ref_max_abs": float(gr.abs().max()),
        **contract,
        "agree": agree,
        "kernel_seconds": round(secs, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tree = os.path.basename(os.environ.get("LIGER_TREE", "").rstrip("/")) or "unknown"
    if DEVICE == "cuda" and os.environ.get("TRITON_INTERPRET", "0") != "1":
        backend = torch.cuda.get_device_name(0)
    else:
        backend = f"interpreter-{DEVICE}"
    t0 = time.time()
    results = []
    for cfg in sample_configs(args.n, args.seed):
        try:
            results.append(run_one(cfg))
        except Exception as e:  # an exception on an in-scope config is a disagreement
            results.append({**cfg, "agree": False, "error": f"{type(e).__name__}: {e}"})
    bad = [r for r in results if not r["agree"]]
    coverage = {
        "reductions": sorted({r["reduction"] for r in results}),
        "dtype_pairs": sorted({f'{r["p_dtype"]}/{r["q_dtype"]}' for r in results}),
        "with_labels": sum(r["labels"] for r in results),
        "with_ignored_rows": sum(1 for r in results if r.get("n_kept_rows", r["BT"]) < r["BT"]),
        "ignore_indices": sorted({r["ignore_index"] for r in results if r["labels"]}),
        "V_values": sorted({r["V"] for r in results}),
        "BT_values": sorted({r["BT"] for r in results}),
        "layouts_p": sorted({r["p_layout"] for r in results}),
        "layouts_q": sorted({r["q_layout"] for r in results}),
        "upstreams": sorted({r["upstream"] for r in results}),
        "q_requires_grad": sum(r["q_requires_grad"] for r in results),
        "exact_ties_total": sum(r.get("n_exact_ties", 0) for r in results),
    }
    summary = {
        "kernel": "tvd",
        "tree": tree,
        "backend": backend,
        "torch": torch.__version__,
        "n_configs": len(results),
        "n_agree": len(results) - len(bad),
        "n_disagree": len(bad),
        "max_loss_ratio": max((r.get("loss_ratio", math.inf) for r in results), default=None),
        "max_grad_ratio": max((r.get("grad_ratio", math.inf) for r in results), default=None),
        "disagreements": [
            {k: r.get(k) for k in ("i", "BT", "V", "reduction", "p_dtype", "q_dtype", "labels", "n_exact_ties",
                                   "loss_ratio", "grad_ratio", "shape_ok", "loss_dtype_float32",
                                   "grad_dtype_is_p_dtype", "q_grad_none", "inputs_unmodified", "error")}
            for r in bad
        ],
        "coverage": coverage,
        "tolerances": {"loss_rtol": 1e-5, "grad_rtol_float32_p": 1e-5, "grad_rtol_bfloat16_p": 2e-2,
                       "atol": "1e-6 * max|ref|"},
        "args": vars(args),
        "seconds": round(time.time() - t0, 1),
    }
    summary = strict(summary)
    results = strict(results)
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage"}, allow_nan=False))
    data = {"kernel": "tvd", "runs": {}}
    if OUT_JSON.exists():
        try:
            data = json.loads(OUT_JSON.read_text())
        except json.JSONDecodeError:
            pass
    key = tree if (args.n, args.seed) == (30, 0) else f"{tree}:n{args.n}:seed{args.seed}"
    data.setdefault("runs", {})[key] = {**summary, "configs": results}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(strict(data), indent=1, allow_nan=False) + "\n")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
