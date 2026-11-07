"""conformance_check: does a hand-written model of the kernel agree with the real kernel?

A Z3 proof (UNSAT for "model != reference derivative") only says something about the kernel if the model it was
written against actually behaves like the kernel. This check evaluates the model and the kernel on random in-domain
inputs and requires agreement under the tolerance rule on every trial. Generic special values are injected: copied
entries (equal values across same-shaped inputs), zeroed ranges, and degenerate sizes. A model that is wrong only on
boundary cases therefore cannot pass on purely continuous draws.

Model interface (source string or callable):
    def model(inputs: dict) -> dict
  inputs: float64 copies of every tensor argument (the exact values the kernel sees), the non-tensor kwargs, and
          "upstream": the float64 upstream gradient for the output.
  returns any of {"output": tensor, "grad_<arg>": tensor, ...}; at least one "grad_<arg>" key is required.
"""

from __future__ import annotations

import copy
import random
import time

import torch

from .compare import _as_tuple, _ref_copy, grad_compare, reduction_size
from .inputs import build_inputs, clone_like, make_upstream
from .spaces import ARGS, diff_args, sample
from .tolerances import allowed_error


def _compile_model(model):
    if callable(model):
        return model
    ns = {"torch": torch}
    exec(compile(model, "<model>", "exec"), ns)
    if "model" not in ns:
        raise ValueError("model source must define `def model(inputs): ...`")
    return ns["model"]


def _inject(spec: dict, kernel: str, rng: random.Random) -> dict:
    """Generic special-value injection into a sampled spec (not specific to any kernel or defect)."""
    s = copy.deepcopy(spec)
    tens = s["tensors"]
    flo = [k for k, t in tens.items() if t and not ARGS[kernel].get(k, {}).get("integer")]
    for a in flo:
        for b in flo:
            if a < b and tens[a]["shape"] == tens[b]["shape"] and rng.random() < 0.5:
                last = tens[b]["shape"][-1]
                c0 = rng.randrange(0, last)
                c1 = rng.choice([last, rng.randrange(c0, last) + 1])
                tens[b].setdefault("edits", []).append({"cols": [c0, c1], "from": a})
    for a in flo:
        if rng.random() < 0.25:
            last = tens[a]["shape"][-1]
            c0 = rng.randrange(0, last)
            tens[a].setdefault("edits", []).append({"cols": [c0, min(last, c0 + max(1, last // 4))], "value": 0.0})
    return s


def _small(spec: dict, cap: int = 96) -> dict:
    """Keep conformance trials cheap: cap every dimension (shapes stay consistent across arguments)."""
    s = copy.deepcopy(spec)
    for t in s["tensors"].values():
        if t:
            t["shape"] = [min(d, cap) for d in t["shape"]]
            if t.get("init", {}).get("kind") == "randint":
                t["init"]["high"] = min(t["init"]["high"], cap)
    return s


def conformance_check(model, kernel_fn, ref_fn, kernel: str, n_trials: int = 24, seed: int = 0,
                      specs: list | None = None) -> dict:
    t0 = time.time()
    fn = _compile_model(model)
    rng = random.Random(seed)
    trials = []
    specs = specs or [_inject(_small(sample(kernel, rng, seed * 1000 + i)), kernel, rng) for i in range(n_trials)]
    for i, spec in enumerate(specs):
        built = build_inputs(spec, diff_args=diff_args(kernel))
        ref = grad_compare(kernel_fn, ref_fn, built)  # also gives the float32-floor per item
        floor = {it["name"]: it.get("err32", 0.0) for it in ref.get("items", [])}
        args = {k: clone_like(v) if isinstance(v, torch.Tensor) else v for k, v in built.call_args().items()}
        try:
            outs = _as_tuple(kernel_fn(**args))
        except Exception as e:
            trials.append({"trial": i, "agree": False, "reason": f"kernel raised {type(e).__name__}: {e}"[:300]})
            continue
        diff = [k for k, v in args.items() if isinstance(v, torch.Tensor) and v.requires_grad]
        kres = {"output": outs[0]}
        up64 = make_upstream(built.upstream, outs[0], torch.float64, int(spec.get("seed", 0)))
        if diff and outs[0].requires_grad:
            g = make_upstream(built.upstream, outs[0], outs[0].dtype, int(spec.get("seed", 0)))
            for k, gk in zip(diff, torch.autograd.grad(outs[0], [args[k] for k in diff], g, allow_unused=True)):
                kres[f"grad_{k}"] = gk
        minputs = {k: (_ref_copy(v, torch.float64).detach() if isinstance(v, torch.Tensor) else v)
                   for k, v in built.call_args().items()}
        minputs["upstream"] = up64.detach().cpu() if up64.device.type != "cpu" else up64.detach()
        minputs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in minputs.items()}
        try:
            mres = fn(minputs)
        except Exception as e:
            trials.append({"trial": i, "agree": False, "reason": f"model raised {type(e).__name__}: {e}"[:300]})
            continue
        keys = [k for k in mres if k in kres and kres[k] is not None]
        if not any(k.startswith("grad_") for k in keys):
            trials.append({"trial": i, "agree": False, "reason": "model returned no comparable grad_<arg> key",
                           "model_keys": list(mres)})
            continue
        n = reduction_size(built.call_args())
        cmp = {}
        for k in keys:
            T = kres[k].detach().to("cpu", torch.float64)
            M = torch.as_tensor(mres[k], dtype=torch.float64).reshape(T.shape) if torch.as_tensor(mres[k]).numel() == T.numel() else None
            if M is None:
                cmp[k] = {"agree": False, "reason": f"shape {list(torch.as_tensor(mres[k]).shape)} vs kernel {list(T.shape)}"}
                continue
            err = float((T - M).abs().max()) if T.numel() else 0.0
            allowed = allowed_error(kres[k].dtype, n, float(M.abs().max()) if M.numel() else 0.0,
                                    floor.get(k if k != "output" else "output", 0.0))
            cmp[k] = {"agree": err <= allowed, "max_abs_err": err, "allowed": allowed}
        trials.append({"trial": i, "agree": all(c["agree"] for c in cmp.values()), "compared": cmp,
                       "shapes": {k: t["shape"] for k, t in spec["tensors"].items() if t},
                       "kwargs": spec.get("kwargs"), "edits": {k: t.get("edits") for k, t in spec["tensors"].items() if t and t.get("edits")}})
    bad = [t for t in trials if not t["agree"]]
    return {"conformance": "pass" if not bad and trials else "fail", "n_trials": len(trials), "n_disagree": len(bad),
            "first_disagreements": bad[:3], "seconds": round(time.time() - t0, 3)}
