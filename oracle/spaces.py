"""Per-kernel argument metadata, example specs, and samplers over each kernel's documented input domain.

Each sampler draws a spec (oracle.inputs format) from the in-scope domain stated in the kernel's task.md. The
distributions are generic and identical in kind across kernels. Sizes are log-uniform over the documented range,
layouts are mixed, upstream gradients are mixed, and every kernel with logits uses the same logit-scale
distribution (log-uniform on [0.5, 40]). No distribution is tuned toward any particular defect.
"""

from __future__ import annotations

import math
import random

# differentiable: does the kernel's contract give this argument a gradient?   matrix: 2-D (eligible for layout variants)
ARGS = {
    "tvd": {"p": dict(differentiable=True, matrix=True), "q": dict(differentiable=False, matrix=True),
            "shift_labels": dict(differentiable=False, matrix=False, integer=True)},
    "flce": {"input": dict(differentiable=True, matrix=True), "weight": dict(differentiable=True, matrix=True),
             "target": dict(differentiable=False, matrix=False, integer=True),
             "bias": dict(differentiable=True, matrix=False)},
    "fljsd": {"student_input": dict(differentiable=True, matrix=True),
              "student_weight": dict(differentiable=True, matrix=True),
              "teacher_input": dict(differentiable=False, matrix=True),
              "teacher_weight": dict(differentiable=False, matrix=True),
              "shift_labels": dict(differentiable=False, matrix=False, integer=True)},
    "ce": {"input": dict(differentiable=True, matrix=True), "target": dict(differentiable=False, matrix=False, integer=True)},
}

# layouts each kernel's scope allows for its tensor arguments
SCOPE_LAYOUTS = {
    "tvd": ["contiguous", "transposed", "col_strided:2", "row_strided:2", "offset:3"],
    "flce": ["contiguous", "transposed", "col_strided:2", "row_strided:2", "offset:3"],
    "fljsd": ["contiguous", "row_strided:2", "offset:3"],          # unit last-dim stride only
    "ce": ["contiguous", "row_strided:2", "offset:3"],             # unit last-dim stride only
}


def diff_args(kernel: str) -> dict:
    return {a: m["differentiable"] for a, m in ARGS[kernel].items()}


def _logu(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _logu_int(rng, lo, hi):
    return max(lo, min(hi, int(round(_logu(rng, lo, hi)))))


def _layout(rng, kernel, matrix=True):
    opts = SCOPE_LAYOUTS[kernel]
    if not matrix:
        opts = [o for o in opts if o in ("contiguous", "offset:3")] + ["strided"]
    weights = [6 if o == "contiguous" else 1 for o in opts]
    return rng.choices(opts, weights)[0]


def _labels(rng, BT, V, need_kept):
    ignore_index = rng.choice([-100, 0, 1]) if V > 1 else -100
    frac = rng.choice([0.0, rng.uniform(0, 0.5), 1.0 if not need_kept else 0.5])
    spec = {"shape": [BT], "dtype": "int64", "init": {"kind": "randint", "low": 0, "high": max(V, 1),
                                                          "ignore_frac": frac, "ignore_index": ignore_index}}
    return spec, ignore_index


def _upstream(rng, reduced: bool):
    if reduced:
        return rng.choice([{"kind": "ones"}, {"kind": "scalar", "value": round(rng.uniform(-2, 2), 3)}])
    return {"kind": "randn", "std": 1.0, "layout": rng.choice(["contiguous", "contiguous", "strided"])}


def sample(kernel: str, rng: random.Random, seed: int) -> dict:
    """One random spec from the kernel's documented in-scope domain."""
    if kernel == "tvd":
        BT, V = _logu_int(rng, 1, 64), _logu_int(rng, 1, 4096)
        dp, dq = rng.choice([("float32", "float32"), ("bfloat16", "float32"), ("float32", "bfloat16")])
        red = rng.choice(["none", "sum", "mean", "batchmean"])
        def dist():
            return ({"kind": "softmax", "std": round(_logu(rng, 0.1, 10), 3)} if rng.random() < 0.7
                    else {"kind": "randn", "std": 1.0})
        t = {"p": {"shape": [BT, V], "dtype": dp, "init": dist(), "layout": _layout(rng, kernel)},
             "q": {"shape": [BT, V], "dtype": dq, "init": dist(), "layout": _layout(rng, kernel)}}
        kw = {"reduction": red}
        if rng.random() < 0.5:
            lab, ii = _labels(rng, BT, V, need_kept=True)
            t["shift_labels"] = lab
            kw["ignore_index"] = ii
            if red in ("mean", "batchmean"):  # scope: at least one kept row; force row 0 kept
                lab["edits"] = [{"index": [0], "value": (ii + 1) % max(V, 2) if ii >= 0 else 0}]
        return {"tensors": t, "kwargs": kw, "upstream": _upstream(rng, red != "none"), "seed": seed}

    if kernel == "flce":
        BT, H, V = _logu_int(rng, 1, 64), _logu_int(rng, 1, 128), _logu_int(rng, 2, 512)  # V >= 2: see task.md
        ls = _logu(rng, 0.5, 40.0)
        red = rng.choice(["mean", "sum", "none"])
        mode = rng.choices(["all", "frozen_weight", "frozen_bias"], [8, 1, 1])[0]
        lab, ii = _labels(rng, BT, V, need_kept=False)
        t = {"input": {"shape": [BT, H], "dtype": "float32", "init": {"kind": "randn"}, "layout": _layout(rng, kernel)},
             "weight": {"shape": [V, H], "dtype": "float32", "init": {"kind": "randn", "std": ls / math.sqrt(H)},
                        "layout": _layout(rng, kernel), "requires_grad": mode != "frozen_weight"},
             "target": lab}
        if rng.random() < 0.5:
            t["bias"] = {"shape": [V], "dtype": "float32", "init": {"kind": "randn", "std": 0.5},
                         "layout": _layout(rng, kernel, matrix=False), "requires_grad": mode != "frozen_bias"}
        kw = {"ignore_index": ii, "reduction": red,
              "label_smoothing": 0.0 if rng.random() < 0.5 else round(rng.uniform(0, 1), 3),
              "lse_square_scale": 0.0 if rng.random() < 0.6 else float(f"{10 ** rng.uniform(-5, -1):.3g}")}
        return {"tensors": t, "kwargs": kw, "upstream": _upstream(rng, red != "none"), "seed": seed}

    if kernel == "fljsd":
        BT, Hs, Ht, V = _logu_int(rng, 1, 64), _logu_int(rng, 1, 128), _logu_int(rng, 1, 128), _logu_int(rng, 2, 1024)
        dt = rng.choice(["float32", "bfloat16", "float16"])
        ls_s, ls_t = _logu(rng, 0.5, 40.0), _logu(rng, 0.5, 40.0)
        beta = rng.choices([0.0, 1.0, round(rng.uniform(0, 1), 3)], [15, 15, 70])[0]
        t = {"student_input": {"shape": [BT, Hs], "dtype": dt, "init": {"kind": "randn"}, "layout": _layout(rng, kernel)},
             "student_weight": {"shape": [V, Hs], "dtype": dt, "init": {"kind": "randn", "std": ls_s / math.sqrt(Hs)},
                                "layout": _layout(rng, kernel)},
             "teacher_input": {"shape": [BT, Ht], "dtype": dt, "init": {"kind": "randn"}, "layout": _layout(rng, kernel)},
             "teacher_weight": {"shape": [V, Ht], "dtype": dt, "init": {"kind": "randn", "std": ls_t / math.sqrt(Ht)},
                                "layout": _layout(rng, kernel)}}
        kw = {"jsd_beta": beta, "temperature": round(_logu(rng, 0.5, 4.0), 3),
              "accum_dtype": rng.choice([None, "float32"])}
        if rng.random() < 0.5:
            lab, ii = _labels(rng, BT, V, need_kept=False)
            t["shift_labels"] = lab
            kw["ignore_index"] = ii
        return {"tensors": t, "kwargs": kw, "upstream": _upstream(rng, True), "seed": seed}

    if kernel == "ce":
        BT, V = _logu_int(rng, 1, 64), _logu_int(rng, 2, 4096)
        dt = rng.choice(["float32", "float16"])
        lab, ii = _labels(rng, BT, V, need_kept=False)
        red = rng.choice(["mean", "sum", "none"])
        t = {"input": {"shape": [BT, V], "dtype": dt, "init": {"kind": "randn", "std": round(_logu(rng, 0.5, 40.0), 3)},
                       "layout": _layout(rng, kernel), "requires_grad": rng.random() < 0.8},
             "target": lab}
        kw = {"ignore_index": ii, "reduction": red,
              "label_smoothing": 0.0 if rng.random() < 0.5 else round(rng.uniform(0, 1), 3),
              "lse_square_scale": 0.0 if rng.random() < 0.6 else float(f"{10 ** rng.uniform(-5, -1):.3g}")}
        return {"tensors": t, "kwargs": kw, "upstream": _upstream(rng, red != "none"), "seed": seed}
    raise KeyError(kernel)


def example_spec(kernel: str) -> dict:
    """A small, ordinary in-scope call (a starting point for writing specs)."""
    ex = {
        "tvd": {"tensors": {"p": {"shape": [4, 16], "dtype": "float32", "init": {"kind": "softmax"}},
                            "q": {"shape": [4, 16], "dtype": "float32", "init": {"kind": "softmax"}}},
                "kwargs": {"reduction": "batchmean"}},
        "flce": {"tensors": {"input": {"shape": [8, 16], "dtype": "float32"},
                             "weight": {"shape": [32, 16], "dtype": "float32", "init": {"kind": "randn", "std": 0.25}},
                             "target": {"shape": [8], "dtype": "int64", "init": {"kind": "randint", "low": 0, "high": 32}}},
                 "kwargs": {"reduction": "mean"}},
        "fljsd": {"tensors": {"student_input": {"shape": [8, 16], "dtype": "float32"},
                              "student_weight": {"shape": [32, 16], "dtype": "float32", "init": {"kind": "randn", "std": 0.25}},
                              "teacher_input": {"shape": [8, 16], "dtype": "float32"},
                              "teacher_weight": {"shape": [32, 16], "dtype": "float32", "init": {"kind": "randn", "std": 0.25}}},
                  "kwargs": {"jsd_beta": 0.5}},
        "ce": {"tensors": {"input": {"shape": [8, 32], "dtype": "float32"},
                           "target": {"shape": [8], "dtype": "int64", "init": {"kind": "randint", "low": 0, "high": 32}}},
               "kwargs": {"reduction": "mean"}},
    }[kernel]
    ex.setdefault("upstream", {"kind": "ones"})
    ex.setdefault("seed", 0)
    return ex
