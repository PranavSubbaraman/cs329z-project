"""HARNESS-ONLY repro for pair P4_fljsd_bf16 (issue #1432, fix PR #1433, kernel fljsd).

Bug: `fused_linear_jsd_forward` projects with `(x @ w.t()).to(torch.float32)`. For bfloat16 inputs the matmul output
is rounded to bfloat16 before the cast, so the "float32" logits carry bf16 rounding error. The JSD gradient is a
difference of nearly equal distributions, so that rounding becomes a several-percent error in grad_input and
grad_weight, while the scalar loss barely moves. The fix requests a float32 GEMM output
(`torch.mm(x, w.t(), out_dtype=torch.float32)`), but ONLY on CUDA with compute capability >= 8, torch >= 2.8 and
same-dtype fp16/bf16 operands. Everywhere else (CPU, the Triton interpreter, older GPUs) the fixed commit keeps the
old expression, so on CPU the bug manifests at BOTH commits; the pair can only be told apart on an Ampere+ GPU.

Setup (the bf16 rows of the fix PR's accuracy table): BT=256, H=1024, V=32000; randn inputs; weights
randn * (std / sqrt(H)); (jsd_beta, logit std) in {(0.0, 30), (0.5, 10), (1.0, 30)}; temperature 1; no labels;
upstream gradient 1 (plain .backward()). Inputs are drawn on CPU from a seeded generator, then moved to the device.
Ground truth: float64 autograd of the same loss on the same (exactly upcast) bf16 values, computed inline.

Pass/fail criterion: rel = max|g_liger - g_ref| / max|g_ref| for g in {grad_input, grad_weight};
  bug_manifest = (max of rel over the 3 configs and both gradients) > 0.02.
Expected: float32-output projection <= ~0.5-0.7% (simulation harness_only/phase1/P4_fljsd_bf16_sim.json and the PR's
table); cast-after-matmul >= ~7% at std 30 for jsd_beta 0 and 1 (simulation with correctly rounded bf16 GEMM
output; the PR measured 10-23% on its GPU). jsd_beta=0.5 at std 10 is reported but is not needed to cross the bar.

Exit 1 if the bug manifests, 0 if not, 2 on error (exception or non-finite values). Prints exactly one JSON line.
Run:     scripts/harness/in_tree.sh <sha> .venv/bin/python harness_only/repros/P4_fljsd_bf16.py
On GPU:  TRITON_INTERPRET=0 scripts/harness/in_tree.sh <sha> .venv/bin/python harness_only/repros/P4_fljsd_bf16.py
"""

import json
import math
import os
import sys
import time

import torch

PAIR_ID = "P4_fljsd_bf16"
THRESHOLD = 0.02
BT, H, V = 256, 1024, 32000
CONFIGS = [(0.0, 30.0), (0.5, 10.0), (1.0, 30.0)]  # (jsd_beta, logit std)
SEED = 0


def make_inputs(std, device):
    g = torch.Generator().manual_seed(SEED)
    xs = torch.randn(BT, H, generator=g)
    xt = torch.randn(BT, H, generator=g)
    ws = torch.randn(V, H, generator=g) * (std / H**0.5)
    wt = torch.randn(V, H, generator=g) * (std / H**0.5)
    return [t.to(torch.bfloat16).to(device) for t in (xs, ws, xt, wt)]


def reference_f64(xs, ws, xt, wt, beta):
    xs = xs.double().requires_grad_(True)
    ws = ws.double().requires_grad_(True)
    log_q = torch.log_softmax(xs @ ws.t(), dim=-1)
    log_p = torch.log_softmax(xt.double() @ wt.double().t(), dim=-1)
    if beta == 0.0:
        per_row = (log_p.exp() * (log_p - log_q)).sum(-1)
    elif beta == 1.0:
        per_row = (log_q.exp() * (log_q - log_p)).sum(-1)
    else:
        log_m = torch.logaddexp(log_p + math.log(beta), log_q + math.log1p(-beta))
        per_row = beta * (log_p.exp() * (log_p - log_m)).sum(-1) + (1 - beta) * (log_q.exp() * (log_q - log_m)).sum(-1)
    loss = per_row.sum() / BT
    loss.backward()
    return loss.detach(), xs.grad, ws.grad


def rel_max(a, b):
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


def rel_fro(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tree = os.path.basename(os.environ.get("LIGER_TREE", "unknown"))
    if device == "cuda":
        backend = torch.cuda.get_device_name(0)
    else:
        backend = "interpreter-cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cpu"
    out = {"pair_id": PAIR_ID, "tree": tree, "backend": backend, "bug_manifest": None, "metrics": {}}
    m = out["metrics"]
    try:
        import liger_kernel.functional  # noqa: F401  (declares dispatcher op locations)
        import liger_kernel.ops.fused_linear_jsd as fl
        from liger_kernel.backends.dispatch import available_backends
        from liger_kernel.transformers.functional import liger_fused_linear_jsd

        cc = torch.cuda.get_device_capability(0) if device == "cuda" else None
        torch_ver = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
        m.update({
            "threshold": THRESHOLD,
            "shape": {"BT": BT, "H": H, "V": V},
            "dtype": "bfloat16",
            "seed": SEED,
            "torch": torch.__version__,
            "triton_interpret": os.environ.get("TRITON_INTERPRET"),
            "cuda_capability": list(cc) if cc else None,
            "float32_output_projection_supported_here": bool(device == "cuda" and cc[0] >= 8 and torch_ver >= (2, 8)),
            "tree_has_project_to_float32": hasattr(fl, "_project_to_float32"),
            "jsd_loss_and_grad_backends": available_backends("jsd_loss_and_grad"),
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction if device == "cuda" else None
            ),
            "configs": [],
        })
        worst = 0.0
        finite = True
        for beta, std in CONFIGS:
            t0 = time.time()
            xs, ws, xt, wt = make_inputs(std, device)
            xs_l = xs.detach().clone().requires_grad_(True)
            ws_l = ws.detach().clone().requires_grad_(True)
            loss = liger_fused_linear_jsd(
                student_input=xs_l, student_weight=ws_l, teacher_input=xt, teacher_weight=wt, jsd_beta=beta
            )
            loss.backward()
            t_lib = time.time() - t0
            l_ref, gi_ref, gw_ref = reference_f64(xs, ws, xt, wt, beta)
            ok = bool(torch.isfinite(loss) and torch.isfinite(xs_l.grad).all() and torch.isfinite(ws_l.grad).all())
            finite = finite and ok
            row = {
                "jsd_beta": beta,
                "logit_std": std,
                "grad_input_rel": rel_max(xs_l.grad, gi_ref),
                "grad_weight_rel": rel_max(ws_l.grad, gw_ref),
                "grad_input_rel_fro": rel_fro(xs_l.grad, gi_ref),
                "grad_weight_rel_fro": rel_fro(ws_l.grad, gw_ref),
                "loss": loss.item(),
                "loss_ref": l_ref.item(),
                "loss_rel": abs(loss.item() - l_ref.item()) / max(abs(l_ref.item()), 1e-300),
                "finite": ok,
                "seconds_library": round(t_lib, 2),
            }
            m["configs"].append(row)
            worst = max(worst, row["grad_input_rel"], row["grad_weight_rel"])
            print(f"[{PAIR_ID}] beta={beta} std={std}: grad_input_rel={row['grad_input_rel']:.4%} "
                  f"grad_weight_rel={row['grad_weight_rel']:.4%} loss_rel={row['loss_rel']:.2e} ({t_lib:.1f}s)",
                  file=sys.stderr, flush=True)
            del xs, ws, xt, wt, xs_l, ws_l, gi_ref, gw_ref
        m["max_grad_rel"] = worst
        if not finite:
            m["error"] = "non-finite loss or gradient from liger"
            print(json.dumps(out))
            return 2
        out["bug_manifest"] = bool(worst > THRESHOLD)
        print(json.dumps(out))
        return 1 if out["bug_manifest"] else 0
    except Exception as e:  # noqa: BLE001
        m["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(out))
        return 2


if __name__ == "__main__":
    sys.exit(main())
