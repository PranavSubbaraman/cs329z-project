"""HARNESS-ONLY **EMULATION** for pair P4_fljsd_bf16: the FIXED commit's library code run on CPU with ONE function
monkeypatched at runtime (no file is edited). This is NOT an unmodified library run.

`liger_kernel.ops.fused_linear_jsd._project_to_float32` takes its `torch.mm(x, w.t(), out_dtype=torch.float32)`
branch only on CUDA sm_80+; on CPU it falls back to the old cast-after-matmul expression, and aten::mm.dtype is not
implemented on CPU. Here it is replaced, for same-dtype bf16/fp16 operands, by `x.float() @ w.float().t()`, which is
numerically equivalent to the out_dtype GEMM (bf16 x bf16 products are exact in fp32; fp32 accumulation either way;
only summation order differs). Everything else (chunking, fp32 log-softmax, the Triton `_jsd_kernel` under
TRITON_INTERPRET=1, the bf16 grad matmuls) is the fixed commit's own code. The repro's own main() and criterion are
then evaluated unchanged.

Run (fixed tree only):
  scripts/harness/in_tree.sh ed1f856d .venv/bin/python harness_only/phase1/P4_fljsd_bf16_patched_projection.py
Writes harness_only/phase1/P4_fljsd_bf16_patched_projection.json
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "harness_only", "phase1", "P4_fljsd_bf16_patched_projection.json")


def main():
    import liger_kernel.ops.fused_linear_jsd as fl

    if not hasattr(fl, "_project_to_float32"):
        print("this tree has no _project_to_float32 (run on the fixed tree)", file=sys.stderr)
        return 2
    original = fl._project_to_float32
    calls = {"emulated": 0, "fallback": 0}

    def emulated(inputs, weight):
        if inputs.dtype in (torch.float16, torch.bfloat16) and weight.dtype == inputs.dtype:
            calls["emulated"] += 1
            return inputs.float() @ weight.float().t()
        calls["fallback"] += 1
        return original(inputs, weight)

    fl._project_to_float32 = emulated
    spec = importlib.util.spec_from_file_location("repro", os.path.join(ROOT, "harness_only", "repros",
                                                                       "P4_fljsd_bf16.py"))
    repro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro)
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        code = repro.main()
    secs = time.time() - t0
    result = {
        "kind": ("EMULATION: fixed-commit library on CPU with _project_to_float32 monkeypatched to a float32-output "
                 "projection (its CUDA sm_80+ branch); not an unmodified library run"),
        "repro_exit_code": code,
        "repro_stdout_json": json.loads(buf.getvalue().strip().splitlines()[-1]),
        "projection_calls": calls,
        "seconds": round(secs, 1),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
