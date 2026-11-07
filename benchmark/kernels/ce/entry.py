"""Entry point under test. Same signature and outputs as reference.reference()."""


def kernel(
    input,
    target,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
    lse_square_scale=0.0,
):
    from liger_kernel.transformers.functional import liger_cross_entropy

    return liger_cross_entropy(
        input=input,
        target=target,
        weight=None,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        lse_square_scale=lse_square_scale,
        softcap=None,
        return_z_loss=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    )
