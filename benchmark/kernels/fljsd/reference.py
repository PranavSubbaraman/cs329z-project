"""Plain-PyTorch reference for the fused linear generalized Jensen-Shannon divergence loss.

Same signature and output structure as entry.kernel(). Differentiable through autograd with respect to
student_input and student_weight; teacher_input and teacher_weight are treated as constants. All arithmetic runs
in the dtype of the inputs, so calling it with float64 tensors gives a float64 result and float64 gradients.
"""

import math

import torch


def reference(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels=None,
    jsd_beta=0.5,
    ignore_index=-100,
    temperature=1.0,
    accum_dtype=None,
):
    # accum_dtype only selects the dtype of an internal buffer of the kernel; the value being computed does not
    # depend on it.
    del accum_dtype

    teacher_input = teacher_input.detach()
    teacher_weight = teacher_weight.detach()

    student_logits = (student_input @ student_weight.t()) / temperature  # (BT, V)
    teacher_logits = (teacher_input @ teacher_weight.t()) / temperature  # (BT, V)
    log_q = torch.log_softmax(student_logits, dim=-1)  # student log-probabilities
    log_p = torch.log_softmax(teacher_logits, dim=-1)  # teacher log-probabilities

    beta = float(jsd_beta)
    if beta == 0.0:
        # forward KL: KL(P || Q)
        per_token = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    elif beta == 1.0:
        # reverse KL: KL(Q || P)
        per_token = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
    else:
        # M = beta * P + (1 - beta) * Q, formed in log space
        log_m = torch.logaddexp(log_p + math.log(beta), log_q + math.log1p(-beta))
        per_token = beta * (log_p.exp() * (log_p - log_m)).sum(dim=-1) + (1.0 - beta) * (
            log_q.exp() * (log_q - log_m)
        ).sum(dim=-1)

    if shift_labels is None:
        n_rows = per_token.shape[0]
        total = per_token.sum()
    else:
        keep = shift_labels != ignore_index
        n_rows = int(keep.sum().item())
        total = torch.where(keep, per_token, torch.zeros_like(per_token)).sum()
    # n_rows == 0 (every row ignored) gives a loss of 0 and zero gradients
    return total / max(n_rows, 1)
