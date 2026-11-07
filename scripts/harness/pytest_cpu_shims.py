"""HARNESS-ONLY pytest plugin: minimal shims so Liger-Kernel's own test files can be collected and run on CPU
under TRITON_INTERPRET=1. It never touches library source; it only adds CUDA-only torch APIs that the *tests*
call at module level and that torch.cpu lacks. Every shim is listed in NOTES.md.

Load with:  python -m pytest -p pytest_cpu_shims ...   (scripts/harness on PYTHONPATH)
"""
import os
import types

import torch

_RAM = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

if not hasattr(torch.cpu, "get_device_properties"):
    # test_*.py files do getattr(torch, infer_device()).get_device_properties(0).total_memory to pick shapes.
    torch.cpu.get_device_properties = lambda device=0: types.SimpleNamespace(
        name="cpu", total_memory=_RAM, major=0, minor=0, multi_processor_count=os.cpu_count())
