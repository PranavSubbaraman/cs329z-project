"""Plain-PyTorch reference for the TVD loss described in task.md.

Same signature and output structure as entry.kernel(). Its float64 autograd is the trusted oracle.
"""
import torch

_REDUCTIONS = ("none", "sum", "mean", "batchmean")


def reference(p, q, shift_labels=None, reduction="batchmean", ignore_index=-100):
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    BT, V = p.shape
    q = q.detach()  # constant target: no gradient flows to q

    tvd = 0.5 * torch.abs(p - q)  # (BT, V)

    if shift_labels is None:
        n_rows = BT
    else:
        keep = shift_labels != ignore_index  # (BT,)
        tvd = torch.where(keep.unsqueeze(1), tvd, torch.zeros_like(tvd))
        n_rows = int(keep.sum().item())

    if reduction == "none":
        return tvd
    total = tvd.sum()
    if reduction == "sum":
        return total
    if n_rows == 0:
        raise ValueError(f"reduction={reduction!r} needs at least one row whose label is not ignore_index")
    if reduction == "mean":
        return total / (n_rows * V)
    return total / n_rows  # batchmean
