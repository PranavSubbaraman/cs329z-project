"""Print (and optionally save) the execution environment: backend, device, torch/triton versions.
Every result file embeds this record.   python scripts/env_report.py [out.json]"""
import json
import os
import platform
import sys


def env_record() -> dict:
    import torch
    try:
        import triton
        triton_version = triton.__version__
    except Exception as e:  # pragma: no cover
        triton_version = f"unavailable ({type(e).__name__})"
    interp = os.environ.get("TRITON_INTERPRET", "0").lower() in ("1", "true", "yes", "on")
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "triton_interpret": interp,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available() and not interp:
        p = torch.cuda.get_device_properties(0)
        rec.update(backend=f"cuda:{p.name}", gpu=p.name, capability=f"{p.major}.{p.minor}",
                   cuda_runtime=torch.version.cuda)
        try:
            import subprocess
            rec["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            pass
    else:
        rec["backend"] = "interpreter-cpu"
    return rec


if __name__ == "__main__":
    r = env_record()
    print(json.dumps(r, indent=2))
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(r, indent=2) + "\n")
