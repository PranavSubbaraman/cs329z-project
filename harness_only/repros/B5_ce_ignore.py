"""HARNESS-ONLY repro for B5_ce_ignore (Liger-Kernel issue #1594, open, no fix merged).

Bug: in liger_cross_entropy_kernel (src/liger_kernel/ops/cross_entropy.py:104-108 at 8c7a5078, unchanged at the
harness clone's HEAD 37607058) the `y == ignore_index` branch stores 0.0 over the whole row of X with no
HAS_GRADIENTS check. HAS_GRADIENTS is `_input.requires_grad`, and the gradient pass for the other rows is guarded
by it, so when no gradient is needed (input.requires_grad is False, with or without torch.no_grad()) the kernel
still overwrites the caller's logits at every row whose target == ignore_index with zeros. The returned loss is
unaffected (checked here against float64 torch).

Numeric pass/fail criterion: for each forward-only call below (input.requires_grad False; a plain call and a call
under torch.no_grad(); reductions "mean", "sum", "none"; float32 logits, BT=8, V=37, 3 targets == -100), count the
rows of the caller's input that are not bitwise equal to a snapshot taken before the call.
    bug_manifest = (total rows changed over all forward-only calls) > 0      (expected: 0)
Exit 1 if the bug manifests, 0 if not, 2 on error. Prints exactly one JSON line.

Recorded separately, NOT counted as this bug (metrics.by_design_grad_write): when input.requires_grad is True the
forward pass stores d(loss)/d(input) into input by design (Liger's in-place gradient trick, cf. issue #272); the
script checks that input then holds the float64 gradient (ignored rows = 0) and that input.grad after backward()
matches it. It also records that this happens when requires_grad is True even under torch.no_grad(), because
HAS_GRADIENTS follows the flag, not grad mode; the guard proposed in #1594 would leave that case unchanged.

Run:  scripts/harness/in_tree.sh 8c7a5078 .venv/bin/python harness_only/repros/B5_ce_ignore.py
"""

import json
import os
import sys

PAIR_ID = "B5_ce_ignore"
TREE = os.path.basename(os.environ.get("LIGER_TREE", "").rstrip("/")) or None


def _backend(device):
    import torch

    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return f"interpreter-{device}"
    return torch.cuda.get_device_name() if device == "cuda" else device


def main():
    import torch
    import torch.nn.functional as F

    import liger_kernel
    from liger_kernel.transformers.functional import liger_cross_entropy

    tree_env = os.environ.get("LIGER_TREE")
    if tree_env and not os.path.realpath(liger_kernel.__file__).startswith(os.path.realpath(tree_env)):
        raise RuntimeError(f"liger_kernel imported from {liger_kernel.__file__}, not from LIGER_TREE={tree_env}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BT, V, IGN = 8, 37, -100
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(BT, V, generator=gen, dtype=torch.float32)
    target = torch.randint(0, V, (BT,), generator=gen)
    ignored = torch.tensor([1, 4, 6])
    target[ignored] = IGN
    ign_mask = target == IGN
    logits, target, ign_mask = logits.to(device), target.to(device), ign_mask.to(device)

    metrics = {"device": device, "BT": BT, "V": V, "dtype": "float32", "ignore_index": IGN,
               "n_ignored_rows": int(ign_mask.sum()), "forward_only": {}}
    total_changed = 0
    for mode in ("plain", "no_grad"):
        for reduction in ("mean", "sum", "none"):
            x = logits.clone()  # requires_grad False
            snap = x.clone()
            if mode == "no_grad":
                with torch.no_grad():
                    out = liger_cross_entropy(input=x, target=target, ignore_index=IGN, reduction=reduction)
            else:
                out = liger_cross_entropy(input=x, target=target, ignore_index=IGN, reduction=reduction)
            ref = F.cross_entropy(snap.double(), target, ignore_index=IGN, reduction=reduction)
            changed = (x != snap).any(dim=1)
            n_changed = int(changed.sum())
            total_changed += n_changed
            metrics["forward_only"][f"{mode}/{reduction}"] = {
                "rows_changed": n_changed,
                "ignored_rows_changed": int((changed & ign_mask).sum()),
                "ignored_rows_now_all_zero": int(((x == 0).all(dim=1) & ign_mask).sum()),
                "non_ignored_rows_changed": int((changed & ~ign_mask).sum()),
                "max_abs_change": float((x - snap).abs().max()),
                "loss_max_abs_err_vs_fp64": float((out.double() - ref).abs().max()),
            }
    metrics["rows_changed_total"] = total_changed
    metrics["rows_changed_expected"] = 0

    # control (not part of the criterion): same logits, no ignored targets -> nothing should change
    x = logits.clone()
    snap = x.clone()
    full_target = torch.where(ign_mask, torch.zeros_like(target), target)
    with torch.no_grad():
        liger_cross_entropy(input=x, target=full_target, ignore_index=IGN, reduction="mean")
    metrics["control_no_ignored_targets_rows_changed"] = int((x != snap).any(dim=1).sum())

    # ---- by-design behaviour (NOT this bug): gradient stored into input when requires_grad is True ----
    x64 = logits.double().clone().requires_grad_(True)
    g64 = torch.autograd.grad(F.cross_entropy(x64, target, ignore_index=IGN, reduction="mean"), x64)[0]
    by_design = {}
    for mode in ("grad_mode", "no_grad"):
        x = logits.clone().requires_grad_(True)
        snap = x.detach().clone()
        if mode == "no_grad":
            with torch.no_grad():
                out = liger_cross_entropy(input=x, target=target, ignore_index=IGN, reduction="mean")
        else:
            out = liger_cross_entropy(input=x, target=target, ignore_index=IGN, reduction="mean")
        post = x.detach().clone()
        entry = {
            "rows_changed": int((post != snap).any(dim=1).sum()),
            "input_now_holds_grad_max_abs_err": float((post.double() - g64).abs().max()),
        }
        if mode == "grad_mode":
            out.backward()
            entry["input_grad_max_abs_err_vs_fp64"] = float((x.grad.double() - g64).abs().max())
            entry["input_grad_aliases_input"] = bool(x.grad.data_ptr() == x.data_ptr())
        by_design[mode] = entry
    metrics["by_design_grad_write"] = by_design

    bug = total_changed > 0
    print(json.dumps({"pair_id": PAIR_ID, "tree": TREE, "backend": _backend(device), "bug_manifest": bug,
                      "metrics": metrics}))
    return 1 if bug else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        backend = None
        try:
            import torch

            backend = _backend("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:  # noqa: BLE001
            pass
        print(json.dumps({"pair_id": PAIR_ID, "tree": TREE, "backend": backend, "bug_manifest": None,
                          "metrics": {}, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(2)
