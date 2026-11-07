"""HARNESS-ONLY repro for pair P2_flce_strides (Liger-Kernel PR #1500).

Bug (buggy commit 395d6166): for reduction "mean"/"sum" the fused linear cross entropy computes grad_input and
grad_weight in the forward pass (allocated with torch.zeros_like, so they inherit the layout of input / weight)
and, when the upstream gradient is not exactly 1.0, rescales them in place with element_mul_kernel. That kernel
addresses row r, column j at  base + r * stride(-2) + j , i.e. it assumes a unit column stride. For a transposed
(dense, permuted-stride) input or weight the gradient has stride(-1) != 1, so some elements are scaled several
times and others never: the returned gradients are wrong although the loss is right.

This repro isolates that bug: reduction="mean" only (it does not touch the reduction="none" path of PR #1298).
Setup: float32, BT=24, H=16, V=40, bias, ~20% of targets = -100, input = x0.t().contiguous().t() (stride (1, BT))
and weight = w0.t().contiguous().t() (stride (1, V)), both leaves; (0.5 * loss).backward().
Ground truth: float64 PyTorch autograd of 0.5 * F.cross_entropy(F.linear(x, w, b), y) on the same values.
Control (reported, not part of the criterion): the same values with contiguous input/weight.

Pass/fail: rel_err(t) = max|liger - truth| / max|truth| for grad_input, grad_weight, grad_bias.
bug_manifest = max rel_err over the three gradients (transposed layout) > 1e-3. A correct float32 kernel agrees
to ~1e-6; the bug leaves elements unscaled or over-scaled (errors of order 0.5-1).

Run: scripts/harness/in_tree.sh <commit> .venv/bin/python harness_only/repros/P2_flce_strides.py
Prints one JSON line; exit 1 = bug manifests, 0 = no bug, 2 = error.
"""
import json
import os
import sys

PAIR_ID = "P2_flce_strides"
THRESHOLD = 1e-3
SCALE = 0.5


def main():
    import torch
    import torch.nn.functional as F
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        backend = f"interpreter-{device}"
    else:
        backend = torch.cuda.get_device_name() if device == "cuda" else "cpu"

    BT, H, V = 24, 16, 40
    g = torch.Generator().manual_seed(1500)
    x0 = torch.randn(BT, H, generator=g)
    w0 = torch.randn(V, H, generator=g) * 0.25
    b0 = torch.randn(V, generator=g) * 0.1
    y = torch.randint(0, V, (BT,), generator=g)
    y[torch.randperm(BT, generator=g)[: BT // 5]] = -100

    def liger(transposed):
        if transposed:
            x = x0.to(device).t().contiguous().t().requires_grad_(True)
            w = w0.to(device).t().contiguous().t().requires_grad_(True)
            assert x.stride() == (1, BT) and w.stride() == (1, V)
        else:
            x = x0.clone().to(device).requires_grad_(True)
            w = w0.clone().to(device).requires_grad_(True)
        b = b0.clone().to(device).requires_grad_(True)
        loss = liger_fused_linear_cross_entropy(input=x, weight=w, target=y.to(device), bias=b, reduction="mean")
        (SCALE * loss).backward()
        return [t.detach().cpu().double() for t in (loss, x.grad, w.grad, b.grad)]

    def truth():
        x = x0.double().requires_grad_(True)
        w = w0.double().requires_grad_(True)
        b = b0.double().requires_grad_(True)
        loss = F.cross_entropy(F.linear(x, w, b), y, reduction="mean")
        (SCALE * loss).backward()
        return [t.detach() for t in (loss, x.grad, w.grad, b.grad)]

    def rel_errs(a, r):
        names = ["loss", "grad_input", "grad_weight", "grad_bias"]
        return {n: float((ai - ri).abs().max() / ri.abs().max()) for n, ai, ri in zip(names, a, r)}

    ref = truth()
    err_t = rel_errs(liger(True), ref)
    err_c = rel_errs(liger(False), ref)
    worst = max(err_t["grad_input"], err_t["grad_weight"], err_t["grad_bias"])
    manifest = worst > THRESHOLD
    metrics = {
        "shape_BT_H_V": [BT, H, V],
        "upstream_scale": SCALE,
        "rel_err_transposed_input_and_weight": {k: float(f"{v:.3e}") for k, v in err_t.items()},
        "rel_err_contiguous_control": {k: float(f"{v:.3e}") for k, v in err_c.items()},
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
