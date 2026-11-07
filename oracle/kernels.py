"""Load a task's kernel entry point and plain-PyTorch reference.

A task directory holds task.md, entry.py (def kernel) and reference.py (def reference). Inside an agent sandbox it is
<sandbox>/task; on the harness side it is benchmark/kernels/<kernel>. The kernel name (tvd, flce, fljsd, ce) selects
the argument metadata and input domain in oracle.spaces.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def task_dir() -> Path:
    env = os.environ.get("KV_TASK_DIR")
    if env:
        return Path(env)
    here = Path.cwd()
    for cand in (here / "task", here.parent / "task"):
        if (cand / "entry.py").exists():
            return cand
    raise FileNotFoundError("no task directory: set KV_TASK_DIR or run from the sandbox")


def kernel_name() -> str:
    name = os.environ.get("KV_KERNEL")
    if name:
        return name
    return task_dir().joinpath("kernel_name.txt").read_text().strip()


def _load(path: Path, mod: str):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_task(directory: str | os.PathLike | None = None, entry: str = "entry.py:kernel"):
    """Return (kernel_fn, reference_fn). entry may be 'file.py:function' relative to the task dir."""
    d = Path(directory) if directory else task_dir()
    fname, _, fn = entry.partition(":")
    kernel_fn = getattr(_load(d / fname, "kv_task_entry"), fn or "kernel")
    reference_fn = _load(d / "reference.py", "kv_task_reference").reference
    return kernel_fn, reference_fn
