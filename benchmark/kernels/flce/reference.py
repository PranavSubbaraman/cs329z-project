"""Plain-PyTorch reference for the fused linear + cross-entropy loss (see task.md).

Same signature and output structure as entry.kernel(). Uses only PyTorch, computes in the dtype of the
floating inputs (call it with float64 copies to get the trusted oracle), and is differentiable through
autograd with respect to input, weight and bias. target is treated as a constant.
"""
import torch


def reference(
    input,
    weight,
    target,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    target = target.detach()

    logits = input @ weight.t()  # (BT, V)
    if bias is not None:
        logits = logits + bias
    V = logits.shape[-1]

    valid = target != ignore_index  # (BT,) bool
    safe_target = torch.where(valid, target, torch.zeros_like(target))

    lse = torch.logsumexp(logits, dim=-1)  # (BT,)
    log_probs = logits - lse.unsqueeze(-1)  # (BT, V)
    nll = -log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)  # -log p[target]
    uniform_nll = -log_probs.sum(dim=-1)  # sum over classes of -log p[c]

    ce = (1.0 - label_smoothing) * nll + (label_smoothing / V) * uniform_nll
    z = lse_square_scale * lse * lse
    zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
    per_token = torch.where(valid, ce + z, zero)  # ignored tokens contribute exactly 0

    if reduction == "none":
        return per_token
    total = per_token.sum()
    if reduction == "sum":
        return total
    n_valid = int(valid.sum().item())
    return total / max(n_valid, 1)  # mean over non-ignored tokens; 0 when every token is ignored
