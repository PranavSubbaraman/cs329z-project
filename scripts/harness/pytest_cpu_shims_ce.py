"""HARNESS-ONLY pytest plugin for the ce kernel (load AFTER pytest_cpu_shims):
    python -m pytest ... -p pytest_cpu_shims -p pytest_cpu_shims_ce

Shim (recorded in harness_only/phase1/ce_notes.md): make the Triton CPU interpreter pass Python-float kernel
arguments as fp32 scalar tensors, as compiled Triton does (see triton_interp_float_shim.py). Without it, every
Liger cross-entropy test that uses class weights or softcap fails on the interpreter with
InterpreterError: AttributeError("'float' object has no attribute 'to'"). Library source is not touched.
"""

from triton_interp_float_shim import install

install()
