"""Input specs: a JSON-serializable description of one concrete call, turned into tensors.

The same spec format is used by the agent (counterexamples), by Z3-derived assignments, by random testing and by
replay, so a counterexample from any source can be re-run anywhere.

    spec = {
      "tensors": {
        "<arg>": {
          "shape": [4, 8],
          "dtype": "float32" | "bfloat16" | "float16" | "float64" | "int64",
          "init": {...},                       # how values are drawn (below); default randn
          "edits": [{...}, ...],               # optional exact overrides applied after init (below)
          "requires_grad": true | false,       # default: the argument's differentiability (oracle.spaces)
          "layout": "contiguous" | "transposed" | "row_strided:K" | "col_strided:K" | "offset:K"
        }, ...
      },
      "kwargs": {"reduction": "sum", ...},     # non-tensor arguments, passed through unchanged
      "upstream": {...},                       # gradient fed to backward (below); default ones
      "seed": 0
    }

init kinds (values are generated in float64 on the CPU, then rounded to the tensor's dtype):
  {"kind": "randn", "mean": 0, "std": 1}        {"kind": "uniform", "low": 0, "high": 1}
  {"kind": "softmax", "std": 1}  softmax over the last dim of randn*std      {"kind": "log_softmax", "std": 1}
  {"kind": "constant", "value": c}              {"kind": "values", "values": <nested list>}
  {"kind": "copy", "from": "<other arg>"}       same values as another (already built) tensor
  {"kind": "randint", "low": 0, "high": N, "ignore_frac": f, "ignore_index": v}   (integer tensors)
edits (applied in order):
  {"index": [i, j], "value": v}                 one element
  {"index": [i, j], "from": "<arg>"}            copy one element from another tensor
  {"rows": [r0, r1, ...], "value": v | "from": "<arg>"}      whole rows
  {"cols": [c0, c1], "value": v | "from": "<arg>"}           column range [c0, c1)
upstream kinds:
  {"kind": "ones"}   {"kind": "scalar", "value": g}   {"kind": "randn", "std": 1}   {"kind": "values", "values": [...]}
  optional "layout": "contiguous" | "strided" (non-scalar upstream only)

Reference inputs are made from the kernel's native-dtype tensors, so the reference sees exactly the values the kernel sees.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16, "float64": torch.float64,
          "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _gen(seed: int, name: str) -> torch.Generator:
    h = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:12], 16)
    return torch.Generator().manual_seed(h)


def _values(name: str, t: dict, built: dict, seed: int) -> torch.Tensor:
    shape = tuple(t.get("shape", []))
    init = t.get("init") or {"kind": "randn"}
    if isinstance(init, str):  # shorthand: "init": "randn"
        init = {"kind": init}
    init = dict(init)
    if init.get("kind") == "copy" and "from" not in init:  # accept "source" / tensor-level "from"
        init["from"] = init.get("source") or t.get("from") or t.get("source")
    kind = init.get("kind", "randn")
    if kind == "copy" and init.get("from") not in built:
        raise ValueError(f"{name}: init copy needs 'from' naming an argument defined earlier in 'tensors' "
                         f"(got {init.get('from')!r}; defined so far: {sorted(built)})")
    g = _gen(seed, name)
    if kind == "randn":
        v = torch.randn(shape, generator=g, dtype=torch.float64) * init.get("std", 1.0) + init.get("mean", 0.0)
    elif kind == "uniform":
        lo, hi = init.get("low", 0.0), init.get("high", 1.0)
        v = torch.rand(shape, generator=g, dtype=torch.float64) * (hi - lo) + lo
    elif kind in ("softmax", "log_softmax"):
        z = torch.randn(shape, generator=g, dtype=torch.float64) * init.get("std", 1.0)
        v = z.softmax(-1) if kind == "softmax" else z.log_softmax(-1)
    elif kind == "constant":
        v = torch.full(shape, float(init["value"]), dtype=torch.float64)
    elif kind == "values":
        v = torch.tensor(init["values"], dtype=torch.float64).reshape(shape) if shape else torch.tensor(init["values"], dtype=torch.float64)
    elif kind == "copy":
        v = built[init["from"]].detach().to("cpu", torch.float64).clone().reshape(shape)
    elif kind == "randint":
        v = torch.randint(int(init.get("low", 0)), int(init["high"]), shape, generator=g).to(torch.float64)
        frac = float(init.get("ignore_frac", 0.0))
        if frac > 0:
            mask = torch.rand(shape, generator=g) < frac
            v[mask] = float(init.get("ignore_index", -100))
    else:
        raise ValueError(f"unknown init kind {kind!r} for {name}; use one of randn, uniform, softmax, log_softmax, "
                         "constant, values, copy, randint")
    for e in t.get("edits", []) or []:
        src = built[e["from"]].detach().to("cpu", torch.float64) if "from" in e else None
        if "index" in e:
            idx = tuple(e["index"])
            v[idx] = src[idx] if src is not None else float(e["value"])
        elif "rows" in e:
            r = list(e["rows"])
            v[r] = src[r] if src is not None else float(e["value"])
        elif "cols" in e:
            c0, c1 = e["cols"]
            v[..., c0:c1] = src[..., c0:c1] if src is not None else float(e["value"])
        else:
            raise ValueError(f"bad edit {e} for {name}")
    return v


def place(values: torch.Tensor, dtype: torch.dtype, layout: str, device: str, requires_grad: bool) -> torch.Tensor:
    """Return a leaf tensor with the given values, dtype and memory layout (strides), on device."""
    vals = values.to(dtype)
    shape = tuple(vals.shape)
    layout = layout or "contiguous"
    kind, _, arg = layout.partition(":")
    k = int(arg) if arg else 2
    if kind == "contiguous" or vals.dim() == 0:
        base = torch.empty(shape, dtype=dtype, device=device)
        view = base
    elif kind == "transposed":
        if vals.dim() != 2:
            raise ValueError("transposed layout needs a 2-D tensor")
        base = torch.empty((shape[1], shape[0]), dtype=dtype, device=device)
        view = base.t()
    elif kind == "row_strided":
        base = torch.empty((shape[0] * k,) + shape[1:], dtype=dtype, device=device)
        view = base[::k]
    elif kind == "col_strided":
        base = torch.empty(shape[:-1] + (shape[-1] * k,), dtype=dtype, device=device)
        view = base[..., ::k]
    elif kind == "strided":  # 1-D convenience
        base = torch.empty((shape[0] * k,) + shape[1:], dtype=dtype, device=device)
        view = base[::k]
    elif kind == "offset":
        flat = torch.empty(vals.numel() + k, dtype=dtype, device=device)
        view = flat[k:].view(shape)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    with torch.no_grad():
        view.copy_(vals.to(device))
    leaf = view.detach()
    if requires_grad and dtype.is_floating_point:
        leaf.requires_grad_(True)
    return leaf


@dataclass
class Built:
    tensors: dict
    kwargs: dict
    upstream: dict
    spec: dict
    layouts: dict = field(default_factory=dict)

    def call_args(self) -> dict:
        return {**self.tensors, **self.kwargs}


def build_inputs(spec: dict, device: str | None = None, diff_args: dict | None = None,
                 layout_override: dict | None = None, requires_grad_override: bool | None = None) -> Built:
    """Materialize a spec. diff_args maps arg name -> differentiable (from oracle.spaces) for requires_grad
    defaults. layout_override maps arg -> layout (used by the layout-invariance property)."""
    device = device or spec.get("device") or default_device()
    if device == "auto":
        device = default_device()
    seed = int(spec.get("seed", 0))
    built_vals, out, layouts = {}, {}, {}
    for name, t in (spec.get("tensors") or {}).items():
        if t is None:
            out[name] = None
            continue
        v = _values(name, t, built_vals, seed)
        dtype = DTYPES[t.get("dtype", "float32")]
        rg = t.get("requires_grad", (diff_args or {}).get(name, False))
        if requires_grad_override is not None:
            rg = requires_grad_override and (diff_args or {}).get(name, rg)
        lay = (layout_override or {}).get(name, t.get("layout", "contiguous"))
        layouts[name] = lay
        tens = place(v, dtype, lay, device, bool(rg))
        out[name] = tens
        built_vals[name] = tens
    kwargs = copy.deepcopy(spec.get("kwargs") or {})
    for k, v in list(kwargs.items()):  # JSON can't hold torch dtypes: "float32" -> torch.float32 for *dtype kwargs
        if k.endswith("dtype") and isinstance(v, str):
            kwargs[k] = DTYPES[v]
    return Built(tensors=out, kwargs=kwargs,
                 upstream=copy.deepcopy(spec.get("upstream") or {"kind": "ones"}), spec=spec, layouts=layouts)


def make_upstream(up: dict, like: torch.Tensor, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Upstream gradient for one output, with the output's shape, in the given dtype, on the output's device."""
    kind = up.get("kind", "ones")
    shape = tuple(like.shape)
    if kind == "ones":
        v = torch.ones(shape, dtype=torch.float64)
    elif kind == "scalar":
        v = torch.full(shape, float(up["value"]), dtype=torch.float64)
    elif kind == "randn":
        v = torch.randn(shape, generator=_gen(seed, "__upstream__"), dtype=torch.float64) * up.get("std", 1.0) + up.get("mean", 0.0)
    elif kind == "values":
        v = torch.tensor(up["values"], dtype=torch.float64).reshape(shape)
    elif kind == "zeros":
        v = torch.zeros(shape, dtype=torch.float64)
    else:
        raise ValueError(f"unknown upstream kind {kind!r}")
    lay = up.get("layout", "contiguous")
    if v.dim() == 0 or lay == "contiguous":
        return v.to(device=like.device, dtype=dtype)
    return place(v, dtype, "transposed" if lay == "transposed" else ("row_strided:2" if lay == "strided" else lay),
                 str(like.device), False)


def clone_like(t: torch.Tensor) -> torch.Tensor:
    """Fresh leaf with identical values, dtype, strides and requires_grad (for independent kernel runs)."""
    if t is None or not isinstance(t, torch.Tensor):
        return t
    base = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device)
    with torch.no_grad():
        base.copy_(t.detach())
    base = base.detach()
    if t.requires_grad:
        base.requires_grad_(True)
    return base
