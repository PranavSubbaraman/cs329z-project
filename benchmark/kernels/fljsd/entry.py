"""Entry point under test. Same signature and outputs as reference.reference()."""


def kernel(
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
    from liger_kernel.transformers.functional import liger_fused_linear_jsd

    return liger_fused_linear_jsd(
        student_input=student_input,
        student_weight=student_weight,
        teacher_input=teacher_input,
        teacher_weight=teacher_weight,
        shift_labels=shift_labels,
        jsd_beta=jsd_beta,
        ignore_index=ignore_index,
        temperature=temperature,
        accum_dtype=accum_dtype,
    )
