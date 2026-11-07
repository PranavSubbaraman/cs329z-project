"""Tolerance rule used by every comparison in the oracle (grad_compare, properties, conformance, replay).

THE RULE. A kernel result T (an output or an input gradient, in the kernel's native dtype d) is compared with the
float64 reference R evaluated on exactly the same input values. It passes iff

    max|T - R|  <=  c_out * u(d) * max|R|                  (1) rounding of the stored result to dtype d
                  + c_acc * u32 * log2(2n) * max|R|        (2) float32 accumulation over reductions of length n
                  + K * max|R32 - R|                       (3) data-dependent floor: the error of the same
                                                               reference evaluated in float32
                  + c_out * eta(d) * sqrt(n)               (4) underflow: eta(d) = smallest positive subnormal of d

  * u(d) is the unit roundoff of d: float32 2^-24, bfloat16 2^-8, float16 2^-11 (float64: 2^-53).
  * n is the reduction size, bounded by the largest dimension of any tensor argument (every reduction in these
    ops runs along one tensor dimension).
  * R32 is the reference evaluated with every floating input cast to float32. Term (3) scales automatically
    with conditioning and with reduction size. When R is tiny only because O(1) terms cancel, R32 carries an
    absolute error of order u32 times those terms, so the comparison gets a meaningful absolute floor.
    When R is exactly 0 and R32 is exactly 0 (e.g. a gradient that is identically zero), the rule allows no error.
  * Term (4) is the absolute part of the IEEE rounding model |fl(x) - x| <= u|x| + eta/2, accumulated over up to n
    terms. It matters only for float16, whose results near 1e-5 are subnormal (eta = 2^-24); for bfloat16 and
    float32 eta is below 1e-38, so the term is negligible. It was added after calibration found float16 false alarms
    on a fixed commit (see PREREGISTRATION.md, Deviations).
  * Identically-zero outputs: when an *output* (not a gradient) is exactly 0 in both the float64 and the float32
    reference (e.g. a single-class loss), relative error is undefined and term (3) is 0 by algebraic luck. Only
    then, the allowed error is c_acc * u32 * log2(2n) * max(1, max|input|): float32 accumulation over operands of the
    inputs' magnitude. Gradients never get this floor, so an exactly-zero gradient must still be exactly zero (e.g.
    at ties). Added after the preliminary random run hit it (PREREGISTRATION.md, Deviations).
  * Constants: c_out = 4, c_acc = 4, K = 8. They were frozen in PREREGISTRATION.md after calibration on the fixed
    commits only.

What this encodes is the mixed-precision contract of training kernels: low-precision inputs are exact values;
arithmetic runs in float32; low-precision rounding is allowed only when a result is stored in a low-precision
tensor (term 1). A kernel that rounds an intermediate (for example, logits) to the input dtype is outside this
contract. Its error then exceeds terms (1)-(3) once the rounding is amplified by the computation.

Errors are measured with the max norm over the whole tensor, normalized by max|R|, not elementwise. One badly
wrong element anywhere fails the comparison, and correctly-rounded tiny elements do not.
"""

from __future__ import annotations

import math

import torch

C_OUT = 4.0
C_ACC = 4.0
K_FLOOR = 8.0

UNIT_ROUNDOFF = {
    torch.float64: 2.0**-53,
    torch.float32: 2.0**-24,
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-11,
}
U32 = UNIT_ROUNDOFF[torch.float32]

SMALLEST_SUBNORMAL = {
    torch.float64: 2.0**-1074,
    torch.float32: 2.0**-149,
    torch.bfloat16: 2.0**-133,
    torch.float16: 2.0**-24,
}


def unit_roundoff(dtype: torch.dtype) -> float:
    return UNIT_ROUNDOFF.get(dtype, U32)


def allowed_error(dtype: torch.dtype, n: int, ref_max: float, err32: float) -> float:
    """Right-hand side of the rule for one compared tensor."""
    n = max(int(n), 1)
    return (C_OUT * unit_roundoff(dtype) * ref_max
            + C_ACC * U32 * math.log2(2 * n) * ref_max
            + K_FLOOR * err32
            + C_OUT * SMALLEST_SUBNORMAL.get(dtype, 0.0) * math.sqrt(n))


def zero_output_floor(n: int, input_scale: float) -> float:
    """Allowed error for an output whose float64 and float32 references are both identically zero."""
    return C_ACC * U32 * math.log2(2 * max(int(n), 1)) * max(1.0, float(input_scale))


def describe() -> dict:
    return {"rule": "max|T-R| <= c_out*u(d)*max|R| + c_acc*u32*log2(2n)*max|R| + K*max|R32-R| + c_out*eta(d)*sqrt(n)",
            "c_out": C_OUT, "c_acc": C_ACC, "K": K_FLOOR,
            "unit_roundoff": {str(k).replace("torch.", ""): v for k, v in UNIT_ROUNDOFF.items()}}
