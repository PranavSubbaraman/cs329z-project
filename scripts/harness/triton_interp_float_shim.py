"""HARNESS-ONLY shim for the Triton CPU interpreter (TRITON_INTERPRET=1), triton 3.9.0.

Problem: compiled Triton passes a Python-float kernel argument to the kernel as an fp32 scalar *tensor*
(jit.mangle_type(float) == "fp32"), so kernel code may call methods on it such as `softcap.to(tl.float32)`.
The interpreter's `_implicit_cvt` (triton/runtime/interpreter.py, "TODO: wrap everything in triton tensors")
wraps bools, ints and tensors but leaves floats as plain Python floats, so the same kernel line raises
`InterpreterError: AttributeError("'float' object has no attribute 'to'")`. Liger's cross-entropy kernel hits
this on its class-weight path (`sum_non_ignore_weight.to(tl.float32)`) and its softcap path (`softcap.to(...)`).

Fix: wrap Python floats as fp32 scalar tensors, which is what compiled Triton does. No library source is
touched; nothing changes for bools/ints/tensors/tuples. Call install() before launching kernels.
"""

import numpy as np


def install():
    import triton.language as tl
    from triton.runtime import interpreter as interp

    if getattr(interp._implicit_cvt, "_float_shim", False):
        return
    orig = interp._implicit_cvt

    def _implicit_cvt(arg):
        if isinstance(arg, float):
            handle = interp.TensorHandle(np.array([arg], dtype=np.float32), tl.float32)
            return tl.tensor(handle, tl.float32)
        return orig(arg)

    _implicit_cvt._float_shim = True
    interp._implicit_cvt = _implicit_cvt
