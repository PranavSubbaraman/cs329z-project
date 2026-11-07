"""HARNESS-ONLY **SIMULATION** for pair P4_fljsd_bf16. This is NOT a library run: it never imports liger_kernel.

Purpose: validate, before any GPU is available, the pass/fail threshold used by
harness_only/repros/P4_fljsd_bf16.py. On CPU both commits evaluate the same projection expression (the fix's
float32-output GEMM is CUDA sm_80+ only), so the pair cannot be told apart by running the library here.

What it does: replicates in plain torch the arithmetic of `fused_linear_jsd_forward` (chunking with the
pre-Blackwell chunk budget, fp32 log-softmax / softmax, the Triton `_jsd_kernel` per-element loss and dX formulas,
the logits-side reduction, the downcast of the logits gradient to the input dtype, and the low-precision
grad_input matmul and grad_weight addmm_), with the student/teacher projection done two ways:

  "cast_after_matmul"  (buggy commit everywhere; fixed commit's fallback off CUDA sm_80+):
                        (x @ w.t()).to(torch.float32)   -> GEMM output rounded to bf16/fp16 first
  "float32_output"     (fixed commit on CUDA sm_80+, torch >= 2.8, same-dtype bf16/fp16 operands):
                        torch.mm(x, w.t(), out_dtype=torch.float32). aten::mm.dtype is not implemented on CPU,
                        so it is emulated by x.float() @ w.float().t(): bf16*bf16 and fp16*fp16 products are exact
                        in fp32 and accumulation is fp32 either way, so the two differ only in summation order.
Every other low-precision GEMM (both projections' bf16 output, grad_input, grad_weight addmm_) is emulated the same
way followed by a single round-to-nearest cast to the storage dtype (see lp_mm).

Ground truth is float64 autograd of the same loss on the same (exactly upcast) inputs.
Metric (same as the fix PR's regression test): rel = max|g - g_ref| / max|g_ref|, per gradient.

Run:   .venv/bin/python harness_only/phase1/P4_fljsd_bf16_sim.py [--seeds 5] [--quick]
Writes harness_only/phase1/P4_fljsd_bf16_sim.json
"""

import argparse
import json
import math
import os
import platform
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "harness_only", "phase1", "P4_fljsd_bf16_sim.json")

# Must match harness_only/repros/P4_fljsd_bf16.py
THRESHOLD = 0.02
REPRO_CONFIGS = [  # (dtype, beta, logit_std) -- the bf16 rows of the fix PR's accuracy table
    ("bfloat16", 0.0, 30.0),
    ("bfloat16", 0.5, 10.0),
    ("bfloat16", 1.0, 30.0),
]
CONTEXT_CONFIGS = [  # other rows of the PR table, reported for comparison only
    ("bfloat16", 0.0, 10.0),
    ("float16", 0.0, 30.0),
    ("float16", 1.0, 30.0),
]
PR_TABLE = {  # fix PR #1433 "Accuracy" table (GB10 GPU): grad_input before/after, grad_weight before/after
    ("bfloat16", 0.0, 30.0): (0.1393, 0.0050, 0.1220, 0.0061),
    ("bfloat16", 0.5, 10.0): (0.2290, 0.0038, 0.2360, 0.0062),
    ("bfloat16", 1.0, 30.0): (0.1235, 0.0060, 0.1028, 0.0071),
    ("bfloat16", 0.0, 10.0): (0.0413, 0.0039, 0.0389, 0.0034),
    ("float16", 0.0, 30.0): (0.0125, 0.0006, 0.0109, 0.0008),
    ("float16", 1.0, 30.0): (0.0218, 0.0008, 0.0153, 0.0004),
}


def make_inputs(BT, H, V, dtype, logit_std, seed):
    """Shared input protocol (same draw order as the repro and as the fix PR's test)."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(BT, H, generator=g)
    xt = torch.randn(BT, H, generator=g)
    ws = torch.randn(V, H, generator=g) * (logit_std / H**0.5)
    wt = torch.randn(V, H, generator=g) * (logit_std / H**0.5)
    return [t.to(dtype) for t in (xs, ws, xt, wt)]


def reference_f64(xs, ws, xt, wt, beta, temperature):
    xs = xs.double().requires_grad_(True)
    ws = ws.double().requires_grad_(True)
    zs = (xs @ ws.t()) / temperature
    zt = (xt.double() @ wt.double().t()) / temperature
    lq, lp = torch.log_softmax(zs, -1), torch.log_softmax(zt, -1)
    if beta == 0.0:
        per = (lp.exp() * (lp - lq)).sum(-1)
    elif beta == 1.0:
        per = (lq.exp() * (lq - lp)).sum(-1)
    else:
        lm = torch.logaddexp(lp + math.log(beta), lq + math.log1p(-beta))
        per = beta * (lp.exp() * (lp - lm)).sum(-1) + (1 - beta) * (lq.exp() * (lq - lm)).sum(-1)
    loss = per.sum() / xs.shape[0]
    loss.backward()
    return loss.detach(), xs.grad, ws.grad


def lp_mm(a, b, out_dtype, acc=None):
    """Emulated GEMM on bf16/fp16/fp32 operands: products exact in fp32, fp32 accumulation, one round-to-nearest
    cast to out_dtype (what cuBLAS does for bf16/fp16 GEMMs; equal to CPU torch bf16 GEMM output up to
    summation-order ties). Used for every GEMM because CPU torch's native bf16 GEMM with a long reduction
    dimension is ~50x slower here."""
    r = a.float() @ b.float()
    if acc is not None:
        r = acc.float() + r
    return r.to(out_dtype)


def simulate(xs, ws, xt, wt, beta, temperature, projection):
    """Plain-torch replica of fused_linear_jsd_forward (no labels, accum_dtype=None, upstream grad 1)."""
    dtype = xs.dtype
    BT, H = xs.shape
    V = ws.shape[0]
    inc_factor = -(-V // H)  # chunk_mem_const = 1 (CPU and pre-Blackwell GPUs)
    chunk = 1 << (max(1, -(-BT // inc_factor)) - 1).bit_length()
    chunk = min(chunk, BT)
    grad_input = torch.zeros_like(xs)
    grad_weight = torch.zeros_like(ws)
    total = torch.zeros((), dtype=torch.float32)
    n = BT
    for s in range(0, BT, chunk):
        e = min(s + chunk, BT)
        xs_c, xt_c = xs[s:e], xt[s:e]
        if projection == "cast_after_matmul":
            zs = lp_mm(xs_c, ws.t(), dtype).to(torch.float32)
            zt = lp_mm(xt_c, wt.t(), dtype).to(torch.float32)
        elif projection == "float32_output":
            zs = lp_mm(xs_c, ws.t(), torch.float32)
            zt = lp_mm(xt_c, wt.t(), torch.float32)
        else:
            raise ValueError(projection)
        zs = zs / temperature
        zt = zt / temperature
        X = torch.log_softmax(zs, -1)  # log Q (student)
        Y = torch.log_softmax(zt, -1)  # log P (teacher)
        sq = torch.softmax(zs, -1)
        if beta == 0.0:
            P = torch.exp(Y - Y.max(-1, keepdim=True).values) * torch.exp(Y.max(-1, keepdim=True).values)
            loss = P * (Y - X)
            dX = -P
        elif beta == 1.0:
            Q = torch.exp(X - X.max(-1, keepdim=True).values) * torch.exp(X.max(-1, keepdim=True).values)
            loss = Q * (X - Y)
            dX = loss + Q
        else:
            mx = torch.maximum(X, Y)
            log_M = mx + torch.log(beta * torch.exp(Y - mx) + (1 - beta) * torch.exp(X - mx))
            Q, P = torch.exp(X), torch.exp(Y)
            loss = beta * P * (Y - log_M) + (1 - beta) * Q * (X - log_M)
            dX = (1 - beta) * Q * (X - log_M)
        loss = loss * (1.0 / n)
        dX = dX * (1.0 / n)
        total = total + loss.sum()
        dz = (dX - sq * dX.sum(-1, keepdim=True)) / temperature
        dz = dz.to(dtype)
        grad_input[s:e] = lp_mm(dz, ws, dtype)
        grad_weight = lp_mm(dz.t(), xs_c, dtype, acc=grad_weight)  # addmm_: one rounding of (C + A @ B)
    return total, grad_input, grad_weight


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


def run_config(BT, H, V, dtype_name, beta, std, seed):
    dtype = getattr(torch, dtype_name)
    xs, ws, xt, wt = make_inputs(BT, H, V, dtype, std, seed)
    l_ref, gi_ref, gw_ref = reference_f64(xs, ws, xt, wt, beta, 1.0)
    row = {"dtype": dtype_name, "beta": beta, "logit_std": std, "seed": seed, "BT": BT, "H": H, "V": V}
    for proj in ("cast_after_matmul", "float32_output"):
        l, gi, gw = simulate(xs, ws, xt, wt, beta, 1.0, proj)
        finite = bool(torch.isfinite(gi).all() and torch.isfinite(gw).all() and torch.isfinite(l))
        row[proj] = {
            "grad_input_rel": rel(gi, gi_ref),
            "grad_weight_rel": rel(gw, gw_ref),
            "grad_input_rel_fro": ((gi.double() - gi_ref).norm() / gi_ref.norm()).item(),
            "grad_weight_rel_fro": ((gw.double() - gw_ref).norm() / gw_ref.norm()).item(),
            "loss_rel": abs(l.item() - l_ref.item()) / abs(l_ref.item()),
            "finite": finite,
        }
    row["ref_loss"] = l_ref.item()
    row["ref_grad_input_absmax"] = gi_ref.abs().max().item()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="1 seed, repro configs only")
    ap.add_argument("--shape", default="256,1024,32000", help="BT,H,V (default: the repro's / fix PR's shapes)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resummarize", action="store_true", help="recompute the summary from the rows in --out")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    BT, H, V = (int(v) for v in args.shape.split(","))
    seeds = [0] if args.quick else list(range(args.seeds))
    t0 = time.time()
    if args.resummarize:
        prev = json.load(open(args.out))
        rows, seconds = prev["rows"], prev["summary"].get("seconds")
        seeds = sorted({r["seed"] for r in rows if (r["dtype"], r["beta"], r["logit_std"]) in REPRO_CONFIGS})
    else:
        rows = []
        for cfg in REPRO_CONFIGS:
            for sd in seeds:
                rows.append(run_config(BT, H, V, *cfg, sd))
                r = rows[-1]
                print(f"{cfg} seed={sd} old gi={r['cast_after_matmul']['grad_input_rel']:.4f} "
                      f"gw={r['cast_after_matmul']['grad_weight_rel']:.4f} | new gi={r['float32_output']['grad_input_rel']:.4f} "
                      f"gw={r['float32_output']['grad_weight_rel']:.4f}", flush=True)
        if not args.quick:
            for cfg in CONTEXT_CONFIGS:
                rows.append(run_config(BT, H, V, *cfg, 0))
                r = rows[-1]
                print(f"{cfg} seed=0 (context) old gi={r['cast_after_matmul']['grad_input_rel']:.4f} "
                      f"gw={r['cast_after_matmul']['grad_weight_rel']:.4f} | new gi={r['float32_output']['grad_input_rel']:.4f} "
                      f"gw={r['float32_output']['grad_weight_rel']:.4f}", flush=True)
        seconds = round(time.time() - t0, 1)

    repro_rows = [r for r in rows if (r["dtype"], r["beta"], r["logit_std"]) in REPRO_CONFIGS]

    def row_val(r, proj):  # the larger of the two gradient errors of one config
        return max(r[proj]["grad_input_rel"], r[proj]["grad_weight_rel"])

    def criterion(proj, seed):  # what the repro compares with THRESHOLD: max over its configs and both gradients
        return max(row_val(r, proj) for r in repro_rows if r["seed"] == seed)

    per_cfg_min_old, per_cfg_max_new = {}, {}
    for r in repro_rows:
        k = f"{r['dtype']}/beta={r['beta']}/std={r['logit_std']}"
        per_cfg_min_old[k] = min(per_cfg_min_old.get(k, 1e9), row_val(r, "cast_after_matmul"))
        per_cfg_max_new[k] = max(per_cfg_max_new.get(k, 0.0), row_val(r, "float32_output"))
    crit_old = {sd: criterion("cast_after_matmul", sd) for sd in seeds}
    crit_new = {sd: criterion("float32_output", sd) for sd in seeds}
    min_crit_old, max_crit_new = min(crit_old.values()), max(crit_new.values())
    summary = {
        "kind": "SIMULATION (plain torch, no liger_kernel import); not a library run",
        "threshold": THRESHOLD,
        "shapes": {"BT": BT, "H": H, "V": V},
        "seeds": seeds,
        "criterion_per_seed": {"cast_after_matmul": crit_old, "float32_output": crit_new},
        "min_criterion_cast_after_matmul": min_crit_old,
        "max_criterion_float32_output": max_crit_new,
        "margin_fixed_path": THRESHOLD / max_crit_new,
        "margin_buggy_path": min_crit_old / THRESHOLD,
        "threshold_separates_paths_for_every_seed": bool(max_crit_new < THRESHOLD < min_crit_old),
        "per_config_min_cast_after_matmul": per_cfg_min_old,
        "per_config_max_float32_output": per_cfg_max_new,
        "note": ("criterion = max over the repro's 3 configs and both gradients of max|g-g_ref|/max|g_ref|; "
                 "per-config minima show jsd_beta=0.5/std=10 alone would not clear the bar for seed 0 "
                 "(2.5%), which is why the repro takes the max over configs"),
        "all_finite": all(r[p]["finite"] for r in rows for p in ("cast_after_matmul", "float32_output")),
        "pr_table_for_comparison": {f"{k[0]}/beta={k[1]}/std={k[2]}": dict(zip(
            ["grad_input_before", "grad_input_after", "grad_weight_before", "grad_weight_after"], v))
            for k, v in PR_TABLE.items()},
        "seconds": seconds,
        "host": {"torch": torch.__version__, "machine": platform.machine(), "device": "cpu"},
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
