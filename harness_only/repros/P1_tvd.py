"""HARNESS-ONLY repro for pair P1_tvd (Liger-Kernel issue #1373, fix PR #1374).

Bug: `_tv_distance_kernel` (src/liger_kernel/ops/tvd.py) writes d loss / d p as
`tl.where(p > q, 0.5 * scale, -0.5 * scale)`, a two-way split of the three-way sign. Elements with p == q
therefore get -0.5 * scale instead of the derivative of 0.5 * |p - q| there, which is 0.5 * sign(0) * scale = 0.
The loss is unaffected; only the gradient at exact ties is wrong.

Setup (deterministic, seed 0): p = softmax(randn(8, 512)), float32, and two tie patterns:
  full     q = p.clone(): every element tied (e.g. self-distillation at step 0)
  partial  q = softmax(randn(8, 512)) with columns 256: copied from p (half of each row tied)
Each pattern runs with every reduction. "none" uses a fixed upstream gradient G ~ U[0.5, 1.5) of shape (8, 512),
and the reduced losses use upstream 1.0 from .backward(). Ground truth is torch float64 autograd of the plain
formula 0.5 * |p - q| (sign(0) = 0), reduced the same way.

Metric per case: grad_err = max_ij |g_liger - g_true| / u_ij, where u_ij = 0.5 * scale * |G_ij| is the magnitude
of an untied gradient element. scale is 1 (none, sum), 1/(BT*V) (mean) or 1/BT (batchmean), and G = 1 for the
reduced losses. float32 rounding gives grad_err ~1e-7, and the bug gives grad_err = 1 at every tie.
Pass/fail: the bug manifests iff max grad_err over all cases > 1e-3. loss_err (relative loss error) is recorded
too. It is expected to be ~0 at both commits.

Prints exactly one JSON line. Exit code 1 = bug manifests, 0 = it does not, 2 = error.
Run:  scripts/harness/in_tree.sh <sha-prefix> .venv/bin/python harness_only/repros/P1_tvd.py
"""
import json
import os
import sys

PAIR_ID = "P1_tvd"
THRESHOLD = 1e-3


def main():
    import torch

    from liger_kernel.transformers.tvd import LigerTVDLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and os.environ.get("TRITON_INTERPRET", "0") != "1":
        backend = torch.cuda.get_device_name(0)
    else:
        backend = f"interpreter-{device}"

    BT, V = 8, 512
    g = torch.Generator(device="cpu").manual_seed(0)
    p = torch.randn(BT, V, generator=g).softmax(-1).to(device)
    q_other = torch.randn(BT, V, generator=g).softmax(-1).to(device)
    G_none = (torch.rand(BT, V, generator=g) + 0.5).to(device)

    q_full = p.clone()
    q_partial = q_other.clone()
    q_partial[:, V // 2 :] = p[:, V // 2 :]
    patterns = {"full": q_full, "partial": q_partial}

    scales = {"none": 1.0, "sum": 1.0, "mean": 1.0 / (BT * V), "batchmean": 1.0 / BT}
    cases = {}
    worst = 0.0
    for pat, q in patterns.items():
        tied = p == q
        for red, scale in scales.items():
            # Liger, float32
            x = p.clone().requires_grad_(True)
            out = LigerTVDLoss(reduction=red)(x, q)
            if red == "none":
                out.backward(G_none)
            else:
                out.backward()
            g_liger = x.grad.double()

            # ground truth: float64 autograd of the plain formula
            x64 = p.double().clone().requires_grad_(True)
            d = 0.5 * torch.abs(x64 - q.double())
            if red == "none":
                ref = d
                ref.backward(G_none.double())
            else:
                ref = {"sum": d.sum(), "mean": d.sum() / (BT * V), "batchmean": d.sum() / BT}[red]
                ref.backward()
            g_true = x64.grad

            G = G_none.double() if red == "none" else torch.ones_like(g_true)
            unit = 0.5 * scale * G.abs()
            grad_err = float(((g_liger - g_true).abs() / unit).max())
            loss_err = float((out.detach().double() - ref.detach()).abs().max() / (ref.detach().abs().max() + 1e-30))
            cases[f"{pat}/{red}"] = {
                "n_tied": int(tied.sum()),
                "n_tied_nonzero_grad": int((g_liger[tied] != 0).sum()),
                "max_abs_grad_at_ties": float(g_liger[tied].abs().max()),
                "liger_grad_at_first_tie": float(g_liger[tied][0]),
                "true_grad_at_first_tie": float(g_true[tied][0]),
                "untied_unit_0.5*scale": 0.5 * scale,
                "grad_err": grad_err,
                "loss_err": loss_err,
                "loss": float(out.detach().sum()) if red == "none" else float(out.detach()),
            }
            worst = max(worst, grad_err)

    manifest = worst > THRESHOLD
    result = {
        "pair_id": PAIR_ID,
        "tree": os.path.basename(os.environ.get("LIGER_TREE", "").rstrip("/")),
        "backend": backend,
        "bug_manifest": bool(manifest),
        "metrics": {
            "max_grad_err": worst,
            "threshold": THRESHOLD,
            "shape": [BT, V],
            "dtype": "float32",
            "full_tie_batchmean_grad_at_tie": cases["full/batchmean"]["liger_grad_at_first_tie"],
            "n_tied_nonzero_grad_total": sum(c["n_tied_nonzero_grad"] for c in cases.values()),
            "max_loss_err": max(c["loss_err"] for c in cases.values()),
            "cases": cases,
        },
    }
    print(json.dumps(result))
    return 1 if manifest else 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:  # report errors as one JSON line too
        print(json.dumps({
            "pair_id": PAIR_ID,
            "tree": os.path.basename(os.environ.get("LIGER_TREE", "").rstrip("/")),
            "backend": "unknown",
            "bug_manifest": None,
            "metrics": {},
            "error": f"{type(e).__name__}: {e}",
        }))
        code = 2
    sys.exit(code)
