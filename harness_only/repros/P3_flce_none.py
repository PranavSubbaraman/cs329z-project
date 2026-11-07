"""HARNESS-ONLY repro for pair P3_flce_none (Liger-Kernel PR #1298).

Bug (buggy commit e44a5fe8): with reduction="none" the fused linear cross entropy returns a per-token loss
vector, but its backward ignores the per-token upstream gradient. grad_input, grad_weight and grad_bias are
computed in the forward pass (grad_weight/grad_bias already summed over tokens) and the backward only rescales
them with element_mul_kernel, which reads a single scalar grad_output[0]. Every gradient is therefore
multiplied by grad_output[0] instead of weighting each token's contribution by grad_output[i]. A uniform
upstream gradient (e.g. ones) hides the bug.

This repro isolates that bug: contiguous float32 tensors only (the column-stride bug of PR #1500, also present at
both P3 commits, needs a transposed layout and reduction mean/sum, so it cannot fire here).

Setup: BT=32, H=32, V=64, bias, ~20% of targets = ignore_index (-100), reduction="none",
loss.backward(g) with a fixed non-uniform g = rand(BT) + 0.5. Ground truth: float64 PyTorch autograd of
F.cross_entropy(F.linear(x, w, b), y, reduction="none") on the same values.
Control (reported, not part of the criterion): the same call with g = ones(BT).

Pass/fail: rel_err(t) = max|liger - truth| / max|truth| for t in loss, grad_input, grad_weight, grad_bias.
bug_manifest = max(rel_err over the three gradients, non-uniform g) > 1e-3. A correct float32 kernel agrees to
~1e-6; the bug gives errors of order 0.1-1.

Run: scripts/harness/in_tree.sh <commit> .venv/bin/python harness_only/repros/P3_flce_none.py
Prints one JSON line; exit 1 = bug manifests, 0 = no bug, 2 = error.
"""
import json
import os
import sys

PAIR_ID = "P3_flce_none"
THRESHOLD = 1e-3


def main():
    import torch
    import torch.nn.functional as F
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        backend = f"interpreter-{device}"
    else:
        backend = torch.cuda.get_device_name() if device == "cuda" else "cpu"

    BT, H, V = 32, 32, 64
    g = torch.Generator().manual_seed(1298)
    x0 = torch.randn(BT, H, generator=g) * 0.5
    w0 = torch.randn(V, H, generator=g) * 0.2
    b0 = torch.randn(V, generator=g) * 0.1
    y = torch.randint(0, V, (BT,), generator=g)
    y[torch.randperm(BT, generator=g)[: BT // 5]] = -100
    upstream = torch.rand(BT, generator=g) + 0.5  # non-uniform per-token upstream gradient

    def liger(up):
        x = x0.clone().to(device).requires_grad_(True)
        w = w0.clone().to(device).requires_grad_(True)
        b = b0.clone().to(device).requires_grad_(True)
        loss = liger_fused_linear_cross_entropy(input=x, weight=w, target=y.to(device), bias=b, reduction="none")
        loss.backward(up.to(device))
        return [t.detach().cpu().double() for t in (loss, x.grad, w.grad, b.grad)]

    def truth(up):
        x = x0.double().requires_grad_(True)
        w = w0.double().requires_grad_(True)
        b = b0.double().requires_grad_(True)
        loss = F.cross_entropy(F.linear(x, w, b), y, reduction="none")
        loss.backward(up.double())
        return [t.detach() for t in (loss, x.grad, w.grad, b.grad)]

    def rel_errs(a, r):
        names = ["loss", "grad_input", "grad_weight", "grad_bias"]
        return {n: float((ai - ri).abs().max() / ri.abs().max()) for n, ai, ri in zip(names, a, r)}

    err = rel_errs(liger(upstream), truth(upstream))
    ctrl = rel_errs(liger(torch.ones(BT)), truth(torch.ones(BT)))
    worst = max(err["grad_input"], err["grad_weight"], err["grad_bias"])
    manifest = worst > THRESHOLD
    metrics = {
        "shape_BT_H_V": [BT, H, V],
        "rel_err_nonuniform_upstream": {k: float(f"{v:.3e}") for k, v in err.items()},
        "rel_err_uniform_upstream_control": {k: float(f"{v:.3e}") for k, v in ctrl.items()},
        "max_grad_rel_err": float(f"{worst:.3e}"),
        "threshold": THRESHOLD,
    }
    return manifest, backend, metrics


if __name__ == "__main__":
    tree = os.path.basename(os.environ.get("LIGER_TREE", "unknown"))
    try:
        manifest, backend, metrics = main()
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"pair_id": PAIR_ID, "tree": tree, "backend": None, "bug_manifest": None,
                          "metrics": {"error": f"{type(e).__name__}: {e}"[:500]}}))
        sys.exit(2)
    print(json.dumps({"pair_id": PAIR_ID, "tree": tree, "backend": backend, "bug_manifest": manifest,
                      "metrics": metrics}))
    sys.exit(1 if manifest else 0)
