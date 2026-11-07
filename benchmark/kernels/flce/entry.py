"""Entry point under test. Same signature and outputs as reference.reference()."""


def kernel(
    input,
    weight,
    target,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
):
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

    return liger_fused_linear_cross_entropy(
        input=input,
        weight=weight,
        target=target,
        bias=bias,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
