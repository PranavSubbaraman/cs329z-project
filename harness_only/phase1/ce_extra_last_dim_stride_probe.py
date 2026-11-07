"""HARNESS-ONLY probe of a SECOND, unlabeled CE defect found while validating the ce reference (not B5_ce_ignore).

Defect: in cross_entropy_forward (src/liger_kernel/ops/cross_entropy.py:408-409 and :459 at 8c7a5078, unchanged at
HEAD 37607058) an input with stride(-1) != 1 is replaced by `_input.contiguous()` and the kernel is then launched
with HAS_GRADIENTS=_input.requires_grad. LigerCrossEntropyFunction.forward runs with autograd disabled, so the
contiguous copy has requires_grad == False, the kernel skips the gradient pass, and backward returns the copy,
i.e. the logits themselves (with ignored rows zeroed by the unguarded store), as d(loss)/d(input). Silent: no error.
Introduced together with HAS_GRADIENTS by #894 (5c2a04d, 2025-10-11); before that the gradient was always computed.

Criterion: x = logits.t().contiguous().t() (same values, stride(-1) != 1), requires_grad; liger_cross_entropy(x, t)
.backward(); compare x.grad with float64 torch autograd. defect_manifest = max|x.grad - ref| > 1e-3 * max|ref|.
Exit 1 if it manifests, 0 if not, 2 on error. One JSON line.
Run: scripts/harness/in_tree.sh 8c7a5078 .venv/bin/python harness_only/phase1/ce_extra_last_dim_stride_probe.py
"""

import json
import os
import sys


def main():
    import torch
    import torch.nn.functional as F

    from liger_kernel.transformers.functional import liger_cross_entropy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 37, generator=g)
    t = torch.randint(0, 37, (6,), generator=g)
    t[2] = -100
    logits, t = logits.to(device), t.to(device)

    x64 = logits.double().clone().requires_grad_(True)
    ref = torch.autograd.grad(F.cross_entropy(x64, t, ignore_index=-100), x64)[0]

    metrics = {}
    for layout in ("contiguous", "last_dim_strided"):
        x = logits.clone() if layout == "contiguous" else logits.t().contiguous().t()
        x.requires_grad_(True)
        loss = liger_cross_entropy(input=x, target=t, ignore_index=-100)
        loss.backward()
        zeroed = logits.clone()
        zeroed[t == -100] = 0
        metrics[layout] = {
            "stride": list(x.stride()),
            "loss_abs_err_vs_fp64": float((loss.detach().double() - F.cross_entropy(logits.double(), t)).abs()),
            "grad_max_abs_err_vs_fp64": float((x.grad.double() - ref).abs().max()),
            "ref_grad_max_abs": float(ref.abs().max()),
            "grad_equals_logits_with_ignored_rows_zeroed": bool(torch.equal(x.grad, zeroed)),
        }

    seen = {}

    class _Probe(torch.autograd.Function):  # mechanism, independent of Triton
        @staticmethod
        def forward(ctx, a):
            seen["input_requires_grad"] = a.requires_grad
            seen["contiguous_copy_requires_grad"] = a.contiguous().requires_grad
            return a.sum()

        @staticmethod
        def backward(ctx, go):
            return go

    leaf = torch.randn(3, 4).t().requires_grad_(True)
    _Probe.apply(leaf)
    metrics["inside_autograd_Function.forward"] = seen
    m = metrics["last_dim_strided"]
    bad = m["grad_max_abs_err_vs_fp64"] > 1e-3 * m["ref_grad_max_abs"]
    interp = os.environ.get("TRITON_INTERPRET", "0") == "1"
    print(json.dumps({"probe": "ce_last_dim_stride_grad", "tree": os.path.basename(os.environ.get("LIGER_TREE", "")),
                      "backend": f"interpreter-{device}" if interp else (torch.cuda.get_device_name() if device == "cuda" else device),
                      "defect_manifest": bad, "metrics": metrics}))
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"probe": "ce_last_dim_stride_grad", "defect_manifest": None, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(2)
