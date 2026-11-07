"""Plain-PyTorch reference for the cross-entropy op described in task.md.

Same signature and output structure as entry.kernel(). Differentiable with respect to `input` through
ordinary autograd; `target` is treated as a constant. All arithmetic runs in the dtype of `input`, so a
float64 `input` gives a float64 result and float64 gradients.
"""

import torch


def reference(
    input,
    target,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
    lse_square_scale=0.0,
):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    if input.dim() != 2:
        raise ValueError(f"input must be 2-D (BT, V), got shape {tuple(input.shape)}")

    target = target.detach()
    n_classes = input.shape[-1]

    counted = target != ignore_index
    safe_target = torch.where(counted, target, torch.zeros_like(target))

    lse = torch.logsumexp(input, dim=-1)  # (BT,)
    log_probs = input - lse.unsqueeze(-1)  # (BT, V)
    nll = -log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)  # (BT,)
    uniform_nll = -log_probs.mean(dim=-1)  # (BT,)

    ce_rows = (1.0 - label_smoothing) * nll + label_smoothing * uniform_nll
    z_rows = lse_square_scale * lse * lse

    row_loss = torch.where(counted, ce_rows + z_rows, torch.zeros_like(ce_rows))

    if reduction == "none":
        return row_loss
    if reduction == "sum":
        return row_loss.sum()

    # reduction == "mean": average over the counted rows (0 when no row is counted)
    n_counted = int(counted.sum().item())
    return row_loss.sum() / max(n_counted, 1)
