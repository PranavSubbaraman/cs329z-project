"""HARNESS-ONLY: validate benchmark/kernels/flce/reference.py against the Liger kernel at one commit.

Usage (from anywhere):
  scripts/harness/in_tree.sh <commit-prefix> .venv/bin/python scripts/harness/ref_vs_kernel_flce.py \
      [--layouts all|contiguous] [--n 30] [--seed 0]

Draws --n random configurations over every in-scope option of task.md (dtype x reduction stratified; random
non-power-of-2 sizes, bias on/off, ignore_index (-100 or an in-vocab class) with 0/some/all targets ignored,
label_smoothing, lse_square_scale, upstream gradient (1 or random), which of weight/bias are trainable or a
forward-only call, and -- with --layouts all -- the memory layout of input/weight/bias/target).
Runs entry.kernel() in the native dtype and reference.reference() on float64 copies, compares the loss and the
input/weight/bias gradients (torch.autograd.grad with the drawn upstream gradient) with a normalized max error
max|k - r| / max|r|  against a dtype tolerance (TOL below), checks that no input was modified, prints a summary
JSON line, and merges the run into harness_only/phase1/flce_ref_vs_kernel.json under "<tree8>/<layouts>".
The layout draws use their own RNG stream, so --layouts contiguous gives the same configurations with every
tensor contiguous. Exit code 0 if every configuration agrees, 1 otherwise.
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import random
import sys
import time
import traceback

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
KDIR = ROOT / "benchmark" / "kernels" / "flce"
OUT = ROOT / "harness_only" / "phase1" / "flce_ref_vs_kernel.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ in agent-visible benchmark/
    spec.loader.exec_module(mod)
    sys.dont_write_bytecode = dwb
    return mod


kernel = _load("flce_entry", KDIR / "entry.py").kernel
reference = _load("flce_reference", KDIR / "reference.py").reference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Review fix (2026-11-06): the flce scope is float32 only. bf16/fp16 were dropped because (a) the July commits
# carry a separate low-precision true-class-gradient bug (#1329, fixed between 9ac26bb8 and 395d6166), so P3's
# fixed instance would disagree in scope, and (b) at LM-realistic logit magnitudes bf16/fp16 errors against the
# float64 oracle exceed any fixed tolerance even at the cleanest commit. Both target bugs reproduce in float32.
DTYPES = [torch.float32]
REDUCTIONS = ["mean", "sum", "none"]
# normalized max-error tolerance (kernel vs float64 reference); see nerr() for the absolute floor.
TOL = {torch.float32: 1e-4}
X_LAYOUTS = ["contiguous", "transposed", "row_sliced", "col_strided", "offset"]
B_LAYOUTS = ["contiguous", "strided"]
Y_LAYOUTS = ["contiguous", "strided"]


def backend_name():
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return f"interpreter-{DEVICE}"
    return torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"


def draw_configs(n, seed, layouts):
    rng = random.Random(seed)
    lrng = random.Random(seed + 7919)  # independent stream for layouts
    # which tensors require grad: a shuffled quota (~70% all, ~10% each other mode) so every mode is covered
    k = max(1, n // 10)
    grad_modes = ["frozen_weight"] * k + ["frozen_bias"] * k + ["forward_only"] * k
    grad_modes += ["all"] * max(0, n - len(grad_modes))
    random.Random(seed + 104729).shuffle(grad_modes)
    cfgs = []
    for i in range(n):
        dtype = DTYPES[i % len(DTYPES)]
        reduction = REDUCTIONS[(i // len(DTYPES)) % len(REDUCTIONS)]
        BT = rng.randint(1, 70)
        H = rng.randint(1, 96)
        V = rng.randint(2, 400)
        bias = rng.random() < 0.6
        ignore_index = -100 if rng.random() < 0.6 else rng.randint(0, V - 1)
        u = rng.random()
        ignore_frac = 0.0 if u < 0.35 else (1.0 if u < 0.42 else rng.uniform(0.05, 0.6))
        label_smoothing = 0.0 if rng.random() < 0.4 else rng.uniform(0.0, 1.0)
        lse_square_scale = 0.0 if rng.random() < 0.4 else 10 ** rng.uniform(-5, -1)
        rng.random()  # keeps the stream of the other draws unchanged
        grad_mode = grad_modes[i]
        upstream_kind = "ones" if rng.random() < 0.3 else "random"
        # log-uniform over [0.5, 40]: includes confident, large-logit tokens (review fix: the old [0.5, 4] never sampled them)
        logit_scale = math.exp(rng.uniform(math.log(0.5), math.log(40.0)))
        data_seed = rng.randint(0, 2**31 - 1)
        lay = dict(
            x=lrng.choice(X_LAYOUTS), w=lrng.choice(X_LAYOUTS), b=lrng.choice(B_LAYOUTS), y=lrng.choice(Y_LAYOUTS)
        )
        if layouts == "contiguous":
            lay = dict(x="contiguous", w="contiguous", b="contiguous", y="contiguous")
        cfgs.append(dict(
            idx=i, dtype=str(dtype).replace("torch.", ""), reduction=reduction, BT=BT, H=H, V=V, bias=bias,
            ignore_index=ignore_index, ignore_frac=round(ignore_frac, 4), label_smoothing=round(label_smoothing, 6),
            lse_square_scale=float(f"{lse_square_scale:.6g}"), grad_mode=grad_mode, upstream=upstream_kind,
            logit_scale=round(logit_scale, 4), data_seed=data_seed, layout=lay,
        ))
    return cfgs


def arrange(t, layout, requires_grad):
    """Return a tensor equal to t (same values) with the requested memory layout; requires_grad as asked."""
    t = t.detach()
    if layout == "contiguous":
        out = t.clone().contiguous()
        return out.requires_grad_(requires_grad)
    if layout == "transposed":  # dense, permuted strides; a leaf like weight=param.t()
        out = t.t().contiguous().t() if t.dim() == 2 else t.clone()
        return out.requires_grad_(requires_grad)
    if layout in ("row_sliced", "strided") and t.dim() == 1:
        buf = torch.zeros(2 * t.shape[0], dtype=t.dtype, device=t.device)
        buf[::2] = t
        return buf.requires_grad_(requires_grad)[::2]
    if layout == "row_sliced":
        buf = torch.zeros(2 * t.shape[0], t.shape[1], dtype=t.dtype, device=t.device)
        buf[::2] = t
        return buf.requires_grad_(requires_grad)[::2]
    if layout == "col_strided":
        buf = torch.zeros(t.shape[0], 2 * t.shape[1], dtype=t.dtype, device=t.device)
        buf[:, ::2] = t
        return buf.requires_grad_(requires_grad)[:, ::2]
    if layout == "offset":
        buf = torch.zeros(t.numel() + 3, dtype=t.dtype, device=t.device)
        buf[3:] = t.reshape(-1)
        return buf.requires_grad_(requires_grad)[3:].view(t.shape)
    raise ValueError(layout)


def make_data(c):
    g = torch.Generator().manual_seed(c["data_seed"])
    dtype = getattr(torch, c["dtype"])
    BT, H, V = c["BT"], c["H"], c["V"]
    x = torch.randn(BT, H, generator=g, dtype=torch.float64)
    w = torch.randn(V, H, generator=g, dtype=torch.float64) * (c["logit_scale"] / math.sqrt(H))
    b = torch.randn(V, generator=g, dtype=torch.float64) * 0.5 if c["bias"] else None
    y = torch.randint(0, V, (BT,), generator=g)
    n_ign = int(round(c["ignore_frac"] * BT))
    if n_ign:
        y[torch.randperm(BT, generator=g)[:n_ign]] = c["ignore_index"]
    if c["reduction"] == "none":
        up = torch.ones(BT) if c["upstream"] == "ones" else torch.randn(BT, generator=g, dtype=torch.float64) * 1.5
    else:
        up = torch.tensor(1.0) if c["upstream"] == "ones" else torch.empty(()).uniform_(-3.0, 3.0, generator=g)
    to = dict(device=DEVICE, dtype=dtype)
    return x.to(**to), w.to(**to), (b.to(**to) if b is not None else None), y.to(DEVICE), up.to(DEVICE)


def nerr(k, r, floor=0.0):
    k = k.detach().to(torch.float64)
    r = r.detach().to(torch.float64)
    if k.shape != r.shape:
        return float("inf")
    if not torch.isfinite(k).all():
        return float("inf")
    scale = max(float(r.abs().max()) if r.numel() else 0.0, floor)
    diff = float((k - r).abs().max()) if r.numel() else 0.0
    return diff / scale if scale > 0 else diff


def run_one(c):
    dtype = getattr(torch, c["dtype"])
    x0, w0, b0, y0, up = make_data(c)
    gm = c["grad_mode"]
    rg_x = gm != "forward_only"
    rg_w = gm in ("all", "frozen_bias")
    rg_b = gm in ("all", "frozen_weight")
    lay = c["layout"]
    x = arrange(x0, lay["x"], rg_x)
    w = arrange(w0, lay["w"], rg_w)
    b = arrange(b0, lay["b"], rg_b) if b0 is not None else None
    y = arrange(y0, lay["y"], False) if lay["y"] == "contiguous" else torch.stack([y0, torch.zeros_like(y0)], 1)[:, 0]
    kw = dict(ignore_index=c["ignore_index"], lse_square_scale=c["lse_square_scale"],
              label_smoothing=c["label_smoothing"], reduction=c["reduction"])
    strides = dict(x=list(x.stride()), w=list(w.stride()), b=(list(b.stride()) if b is not None else None),
                   y=list(y.stride()))

    # kernel, native dtype
    k_loss = kernel(input=x, weight=w, target=y, bias=b, **kw)
    wrt_names = [n for n, t, rg in (("input", x, rg_x), ("weight", w, rg_w), ("bias", b, rg_b)) if t is not None and rg]
    wrt_k = [{"input": x, "weight": w, "bias": b}[n] for n in wrt_names]
    k_grads = {}
    if wrt_k:
        gk = torch.autograd.grad(k_loss, wrt_k, up.to(k_loss.dtype), allow_unused=True)
        k_grads = dict(zip(wrt_names, gk))

    # reference, float64 copies
    x64 = x0.detach().to(torch.float64).requires_grad_(rg_x)
    w64 = w0.detach().to(torch.float64).requires_grad_(rg_w)
    b64 = b0.detach().to(torch.float64).requires_grad_(rg_b) if b0 is not None else None
    r_loss = reference(x64, w64, y0, b64, **kw)
    r_grads = {}
    if wrt_names:
        wrt_r = [{"input": x64, "weight": w64, "bias": b64}[n] for n in wrt_names]
        gr = torch.autograd.grad(r_loss, wrt_r, up.to(torch.float64))
        r_grads = dict(zip(wrt_names, gr))

    # Absolute floor (review fix): normalize by max(max|ref|, natural scale) so that values that are tiny only because
    # O(1) terms cancel (e.g. one confidently-correct token) are not judged relative to their own tiny magnitude.
    # Natural scales: per-token loss O(1); dZ entries are bounded by |g|/N (N = kept tokens under mean, else 1), so
    # dX ~ |g|/N * max|W|, dW ~ |g|/N * max|X|, db ~ |g|/N.
    n_keep = int((y0 != c["ignore_index"]).sum())
    per_tok = float(up.abs().max()) / (max(n_keep, 1) if c["reduction"] == "mean" else 1.0)
    floors = {"loss": 1.0 if c["reduction"] != "none" else 1.0,
              "grad_input": per_tok * max(1.0, float(w0.abs().max())),
              "grad_weight": per_tok * max(1.0, float(x0.abs().max())),
              "grad_bias": per_tok}
    errs = {"loss": nerr(k_loss, r_loss, floors["loss"])}
    for n in wrt_names:
        errs[f"grad_{n}"] = float("inf") if k_grads[n] is None else nerr(k_grads[n], r_grads[n], floors[f"grad_{n}"])
    tol = TOL[dtype]
    bad = sorted(n for n, e in errs.items() if not (e <= tol))
    # the op must leave its inputs untouched (values compared after forward + backward)
    modified = [n for n, t, t0 in (("input", x, x0), ("weight", w, w0), ("bias", b, b0), ("target", y, y0))
                if t is not None and not torch.equal(t.detach(), t0)]
    bad += [f"modified_{n}" for n in modified]
    return dict(errors={k: (None if math.isinf(v) else float(f"{v:.3e}")) for k, v in errs.items()},
                inf=[k for k, v in errs.items() if math.isinf(v)], tol=tol, agree=not bad, disagree_on=bad,
                inputs_modified=modified, strides=strides, loss_dtype=str(k_loss.dtype).replace("torch.", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", choices=["all", "contiguous"], default="all")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="JSON file to merge this run into")
    a = ap.parse_args()
    tree = os.path.basename(os.environ.get("LIGER_TREE", "unknown"))
    t0 = time.time()
    results = []
    for c in draw_configs(a.n, a.seed, a.layouts):
        try:
            res = run_one(c)
        except Exception as e:  # a crash counts as a disagreement and is recorded
            res = dict(errors={}, agree=False, disagree_on=["exception"],
                       exception=f"{type(e).__name__}: {str(e)[:300]}", traceback=traceback.format_exc()[-1500:])
        results.append(dict(config=c, **res))
    dis = [r for r in results if not r["agree"]]
    worst = {}
    for r in results:
        d = r["config"]["dtype"]
        for k, v in r.get("errors", {}).items():
            if r["agree"] and v is not None:
                worst.setdefault(d, {}).setdefault(k, 0.0)
                worst[d][k] = max(worst[d][k], v)
    summary = dict(
        kernel="flce", tree=tree, backend=backend_name(), layouts=a.layouts, n_configs=len(results),
        seed=a.seed, n_agree=len(results) - len(dis), n_disagree=len(dis), tolerances={str(k).replace("torch.", ""): v for k, v in TOL.items()},
        max_err_among_agreeing_by_dtype=worst,
        disagreements=[dict(idx=r["config"]["idx"], dtype=r["config"]["dtype"], reduction=r["config"]["reduction"],
                            layout=r["config"]["layout"], grad_mode=r["config"]["grad_mode"],
                            upstream=r["config"]["upstream"], disagree_on=r["disagree_on"],
                            errors=r.get("errors"), exception=r.get("exception")) for r in dis],
        seconds=round(time.time() - t0, 2),
    )
    print(json.dumps(summary))
    if not a.no_save:
        out = pathlib.Path(a.out)
        data = {}
        if out.exists():
            try:
                data = json.loads(out.read_text())
            except Exception:
                data = {}
        data.setdefault("runs", {})[f"{tree[:8]}/{a.layouts}"] = dict(summary=summary, results=results)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=1))
    sys.exit(0 if not dis else 1)


if __name__ == "__main__":
    main()
