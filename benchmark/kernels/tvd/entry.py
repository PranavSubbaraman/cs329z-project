"""Entry point under test. Same signature and outputs as reference.reference()."""


def kernel(p, q, shift_labels=None, reduction="batchmean", ignore_index=-100):
    from liger_kernel.transformers.functional import liger_tvd

    return liger_tvd(
        input=p,
        target=q,
        shift_labels=shift_labels,
        reduction=reduction,
        ignore_index=ignore_index,
    )
