"""Math checks for benchmark/kernels/flce/reference.py (pure PyTorch, no Liger, no Triton).

Run: .venv/bin/python -m pytest tests/test_reference_flce.py -q
"""
import importlib.util
import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

_REF_PATH = pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "kernels" / "flce" / "reference.py"
_spec = importlib.util.spec_from_file_location("flce_reference", _REF_PATH)
_mod = importlib.util.module_from_spec(_spec)
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep benchmark/kernels/ (agent-visible) free of __pycache__
_spec.loader.exec_module(_mod)
sys.dont_write_bytecode = _dwb
reference = _mod.reference

F64 = torch.float64
REDUCTIONS = ["mean", "sum", "none"]


def make_inputs(BT=13, H=7, V=11, bias=True, n_ignored=3, ignore_index=-100, dtype=F64, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BT, H, generator=g, dtype=F64).to(dtype)
    w = (torch.randn(V, H, generator=g, dtype=F64) * scale / math.sqrt(H)).to(dtype)
    b = (torch.randn(V, generator=g, dtype=F64) * 0.5).to(dtype) if bias else None
    y = torch.randint(0, V, (BT,), generator=g)
    if n_ignored:
        idx = torch.randperm(BT, generator=g)[:n_ignored]
        y[idx] = ignore_index
    return x, w, b, y


def leaf(t):
    return None if t is None else t.detach().clone().requires_grad_(True)


def run(fn, x, w, b, y, upstream=None, prepared=False, **kw):
    """Forward + backward with an optional upstream gradient; returns (loss, gx, gw, gb).

    prepared=True: x/w/b already require grad (they may be non-leaf views); otherwise fresh leaves are made.
    """
    if not prepared:
        x, w, b = leaf(x), leaf(w), leaf(b)
    loss = fn(x, w, y, b, **kw) if fn is reference else fn(x, w, b, y, **kw)
    if upstream is None:
        upstream = torch.ones_like(loss)
    wrt = [t for t in (x, w, b) if t is not None]
    grads = list(torch.autograd.grad(loss, wrt, upstream))
    gx, gw = grads[0], grads[1]
    gb = grads[2] if b is not None else None
    return loss.detach(), gx, gw, gb


def torch_builtin(x, w, b, y, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
    return F.cross_entropy(
        F.linear(x, w, b), y, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction
    )


def close(a, b, tol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    return torch.allclose(a, b, atol=tol, rtol=tol)


# ----------------------------------------------------------------------------------------------------------------------
# agreement with torch.nn.functional.cross_entropy (z-loss off)
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ignore_index", [-100, 4])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.7, 1.0])
def test_matches_torch_cross_entropy(reduction, bias, ignore_index, label_smoothing):
    x, w, b, y = make_inputs(bias=bias, ignore_index=ignore_index, seed=1)
    upstream = None
    if reduction == "none":
        upstream = torch.rand(x.shape[0], dtype=F64) * 2 - 0.5
    else:
        upstream = torch.tensor(-1.7, dtype=F64)
    kw = dict(ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction)
    got = run(reference, x, w, b, y, upstream=upstream, **kw)
    exp = run(torch_builtin, x, w, b, y, upstream=upstream, **kw)
    for g, e in zip(got, exp):
        assert close(g, e), (g, e)


def test_matches_torch_large_vocab_chunk_free():
    # bigger, non-power-of-two shapes; reference never chunks, torch materializes logits as well
    x, w, b, y = make_inputs(BT=37, H=19, V=301, n_ignored=9, seed=2, scale=3.0)
    for reduction in REDUCTIONS:
        got = run(reference, x, w, b, y, reduction=reduction, label_smoothing=0.05)
        exp = run(torch_builtin, x, w, b, y, reduction=reduction, label_smoothing=0.05)
        for g, e in zip(got, exp):
            assert close(g, e)


# ----------------------------------------------------------------------------------------------------------------------
# z-loss term
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_z_loss_term_is_scaled_lse_squared(reduction):
    s = 3e-2
    x, w, b, y = make_inputs(seed=3)
    with torch.no_grad():
        base = reference(x, w, y, b, reduction=reduction, label_smoothing=0.2)
        with_z = reference(x, w, y, b, reduction=reduction, label_smoothing=0.2, lse_square_scale=s)
        lse = torch.logsumexp(x @ w.t() + b, dim=-1)
        valid = y != -100
        z = torch.where(valid, s * lse**2, torch.zeros_like(lse))
        expected = {"none": z, "sum": z.sum(), "mean": z.sum() / valid.sum()}[reduction]
    assert close(with_z - base, expected)


# ----------------------------------------------------------------------------------------------------------------------
# gradcheck (float64, smooth everywhere: no kinks in this op)
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("label_smoothing, lse_square_scale", [(0.0, 0.0), (0.15, 1e-2), (1.0, 0.5)])
def test_gradcheck(reduction, bias, label_smoothing, lse_square_scale):
    x, w, b, y = make_inputs(BT=6, H=4, V=5, bias=bias, n_ignored=2, seed=4)
    args = [leaf(x), leaf(w)] + ([leaf(b)] if bias else [])

    def f(*a):
        return reference(
            a[0], a[1], y, a[2] if bias else None, label_smoothing=label_smoothing,
            lse_square_scale=lse_square_scale, reduction=reduction,
        )

    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(f, args, eps=1e-6, atol=1e-6, rtol=1e-4)


# ----------------------------------------------------------------------------------------------------------------------
# closed-form derivative
#   dL_i/dz_ic = (1 + 2 s lse_i) p_ic - (1 - ls) [c == y_i] - ls / V   for non-ignored i, 0 for ignored i
#   mean divides by the number of non-ignored tokens; input/weight/bias grads are the linear-layer pullbacks.
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("ignore_index", [-100, 2])
def test_closed_form_gradients(reduction, ignore_index):
    ls, s = 0.3, 2e-2
    x, w, b, y = make_inputs(BT=17, H=6, V=9, n_ignored=5, ignore_index=ignore_index, seed=5, scale=2.0)
    BT, V = x.shape[0], w.shape[0]
    if reduction == "none":
        up = torch.randn(BT, dtype=F64)
    else:
        up = torch.tensor(0.37, dtype=F64)
    loss, gx, gw, gb = run(
        reference, x, w, b, y, upstream=up, reduction=reduction, label_smoothing=ls, lse_square_scale=s,
        ignore_index=ignore_index,
    )
    z = x @ w.t() + b
    lse = torch.logsumexp(z, -1)
    p = torch.softmax(z, -1)
    valid = (y != ignore_index).to(F64)
    onehot = F.one_hot(torch.where(y == ignore_index, 0, y), V).to(F64)
    G = (1 + 2 * s * lse).unsqueeze(-1) * p - (1 - ls) * onehot - ls / V
    G = G * valid.unsqueeze(-1)
    if reduction == "mean":
        G = G / valid.sum()
    G = G * (up.unsqueeze(-1) if reduction == "none" else up)
    assert close(gx, G @ w)
    assert close(gw, G.t() @ x)
    assert close(gb, G.sum(0))
    # forward closed form
    nll = lse - z.gather(1, torch.where(y == ignore_index, 0, y).unsqueeze(1)).squeeze(1)
    per_tok = ((1 - ls) * nll + ls / V * (V * lse - z.sum(-1)) + s * lse**2) * valid
    expected = {"none": per_tok, "sum": per_tok.sum(), "mean": per_tok.sum() / valid.sum()}[reduction]
    assert close(loss, expected)


# ----------------------------------------------------------------------------------------------------------------------
# upstream-gradient linearity and reduction consistency
# ----------------------------------------------------------------------------------------------------------------------
def test_none_reduction_vector_jacobian_product():
    x, w, b, y = make_inputs(BT=8, H=5, V=7, n_ignored=2, seed=6)
    up = torch.randn(8, dtype=F64)
    _, gx, gw, gb = run(reference, x, w, b, y, upstream=up, reduction="none", label_smoothing=0.1, lse_square_scale=1e-2)
    acc = [torch.zeros_like(gx), torch.zeros_like(gw), torch.zeros_like(gb)]
    for i in range(8):
        e = torch.zeros(8, dtype=F64)
        e[i] = 1.0
        _, gxi, gwi, gbi = run(reference, x, w, b, y, upstream=e, reduction="none", label_smoothing=0.1,
                               lse_square_scale=1e-2)
        for a, gi in zip(acc, (gxi, gwi, gbi)):
            a += up[i] * gi
    assert close(gx, acc[0]) and close(gw, acc[1]) and close(gb, acc[2])


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("c", [0.0, -2.5, 0.5])
def test_scalar_upstream_scales_gradients(reduction, c):
    x, w, b, y = make_inputs(seed=7)
    _, gx1, gw1, gb1 = run(reference, x, w, b, y, reduction=reduction, lse_square_scale=1e-3)
    _, gxc, gwc, gbc = run(reference, x, w, b, y, upstream=torch.tensor(c, dtype=F64), reduction=reduction,
                           lse_square_scale=1e-3)
    assert close(gxc, c * gx1) and close(gwc, c * gw1) and close(gbc, c * gb1)


def test_reductions_consistent():
    x, w, b, y = make_inputs(BT=21, n_ignored=6, seed=8)
    kw = dict(label_smoothing=0.25, lse_square_scale=5e-3)
    with torch.no_grad():
        none = reference(x, w, y, b, reduction="none", **kw)
        total = reference(x, w, y, b, reduction="sum", **kw)
        mean = reference(x, w, y, b, reduction="mean", **kw)
    n_valid = (y != -100).sum()
    assert none.shape == (21,)
    assert total.shape == () and mean.shape == ()
    assert close(none.sum(), total)
    assert close(total / n_valid, mean)
    assert bool((none >= 0).all())  # cross entropy (+ non-negative z-loss) is non-negative


# ----------------------------------------------------------------------------------------------------------------------
# ignore_index semantics
# ----------------------------------------------------------------------------------------------------------------------
def test_ignored_rows_contribute_nothing():
    x, w, b, y = make_inputs(BT=12, n_ignored=4, seed=9)
    keep = y != -100
    for reduction in REDUCTIONS:
        loss, gx, gw, gb = run(reference, x, w, b, y, reduction=reduction, label_smoothing=0.1,
                               lse_square_scale=1e-2)
        assert torch.all(gx[~keep] == 0)
        if reduction == "none":
            assert torch.all(loss[~keep] == 0)
        # dropping the ignored rows gives the same loss and parameter gradients
        loss_k, gx_k, gw_k, gb_k = run(reference, x[keep], w, b, y[keep], reduction=reduction, label_smoothing=0.1,
                                       lse_square_scale=1e-2)
        assert close(loss[keep] if reduction == "none" else loss, loss_k)
        assert close(gx[keep], gx_k) and close(gw, gw_k) and close(gb, gb_k)


def test_ignore_index_inside_vocab():
    x, w, b, y = make_inputs(BT=15, V=6, n_ignored=0, seed=10)
    y[:5] = 3
    loss, *_ = run(reference, x, w, b, y, ignore_index=3, reduction="none")
    assert torch.all(loss[y == 3] == 0) and torch.all(loss[y != 3] > 0)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_all_targets_ignored_gives_zero(reduction):
    x, w, b, _ = make_inputs(seed=11)
    y = torch.full((x.shape[0],), -100)
    loss, gx, gw, gb = run(reference, x, w, b, y, reduction=reduction, label_smoothing=0.3, lse_square_scale=0.1)
    assert torch.all(loss == 0) and torch.isfinite(loss).all()
    assert torch.all(gx == 0) and torch.all(gw == 0) and torch.all(gb == 0)


# ----------------------------------------------------------------------------------------------------------------------
# op-specific identities
# ----------------------------------------------------------------------------------------------------------------------
def test_uniform_logits_known_value():
    BT, H, V, s = 5, 3, 8, 0.1
    x = torch.randn(BT, H, dtype=F64)
    w = torch.zeros(V, H, dtype=F64)
    y = torch.randint(0, V, (BT,))
    for ls in (0.0, 0.4):
        with torch.no_grad():
            loss = reference(x, w, y, None, reduction="none", label_smoothing=ls, lse_square_scale=s)
        assert close(loss, torch.full((BT,), math.log(V) + s * math.log(V) ** 2, dtype=F64))


def test_shift_invariance_without_z_loss():
    # adding the same vector u to every row of weight (and a constant to bias) shifts each row's logits by a
    # constant, which softmax cross entropy ignores; the bias gradient then sums to zero over the vocabulary
    x, w, b, y = make_inputs(seed=12)
    u = torch.randn(1, w.shape[1], dtype=F64)
    for reduction in REDUCTIONS:
        with torch.no_grad():
            l0 = reference(x, w, y, b, reduction=reduction, label_smoothing=0.2)
            l1 = reference(x, w + u, y, b + 1.234, reduction=reduction, label_smoothing=0.2)
        assert close(l0, l1)
        _, _, _, gb = run(reference, x, w, b, y, reduction=reduction, label_smoothing=0.2)
        assert abs(float(gb.sum())) < 1e-12


def test_row_permutation_equivariance():
    x, w, b, y = make_inputs(BT=10, n_ignored=3, seed=13)
    perm = torch.randperm(10)
    l, gx, gw, gb = run(reference, x, w, b, y, reduction="none", upstream=torch.arange(10, dtype=F64))
    lp, gxp, gwp, gbp = run(reference, x[perm], w, b, y[perm], reduction="none",
                            upstream=torch.arange(10, dtype=F64)[perm])
    assert close(l[perm], lp) and close(gx[perm], gxp) and close(gw, gwp) and close(gb, gbp)


def _transposed(t):
    v = t.t().contiguous().t().requires_grad_(True)  # dense leaf with permuted strides
    assert v.stride() == (1, t.shape[0])
    return v


def _strided(t, step=2):
    # non-leaf view with a non-unit last-dim stride into a larger leaf buffer
    shape = list(t.shape)
    shape[-1] *= step
    base = torch.zeros(shape, dtype=t.dtype)
    base[..., ::step] = t
    base.requires_grad_(True)
    v = base[..., ::step]
    assert v.stride()[-1] == step
    return v


def test_layout_invariance():
    # the reference is layout agnostic: transposed / strided views give identical values and gradients
    x, w, b, y = make_inputs(BT=9, H=6, V=10, seed=14)
    ys = torch.stack([y, torch.zeros_like(y)], 1)[:, 0]  # strided target view
    for reduction in REDUCTIONS:
        up = torch.rand(9, dtype=F64) if reduction == "none" else torch.tensor(0.75, dtype=F64)
        base = run(reference, x, w, b, y, upstream=up, reduction=reduction)
        variants = [
            (_transposed(x), _transposed(w), leaf(b), y),
            (_strided(x), leaf(w), _strided(b), ys),
            (leaf(x), _transposed(w), _strided(b), y),
            (_strided(x), _strided(w), None, y),
        ]
        for xx, ww, bb, yy in variants:
            got = run(reference, xx, ww, bb, yy, upstream=up, prepared=True, reduction=reduction)
            exp = base if bb is not None else run(reference, x, w, None, y, upstream=up, reduction=reduction)
            for g, e in zip(got, exp):
                assert close(g, e)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_dtype_follows_input(dtype):
    x, w, b, y = make_inputs(dtype=dtype, seed=15)
    loss, gx, gw, gb = run(reference, x, w, b, y, reduction="mean", label_smoothing=0.1, lse_square_scale=1e-3)
    assert loss.dtype == dtype and gx.dtype == dtype and gw.dtype == dtype and gb.dtype == dtype
    ref64 = run(reference, x.to(F64), w.to(F64), b.to(F64), y, reduction="mean", label_smoothing=0.1,
                lse_square_scale=1e-3)
    tol = {torch.float32: 1e-5, torch.bfloat16: 5e-2, torch.float16: 5e-3, torch.float64: 1e-12}[dtype]
    for g, e in zip((loss, gx, gw, gb), ref64):
        assert float((g.to(F64) - e).abs().max()) <= tol * max(float(e.abs().max()), 1.0)


def test_inputs_not_modified_and_target_constant():
    x, w, b, y = make_inputs(seed=16)
    xc, wc, bc, yc = x.clone(), w.clone(), b.clone(), y.clone()
    run(reference, x, w, b, y, reduction="sum", label_smoothing=0.1, lse_square_scale=1e-2)
    assert torch.equal(x, xc) and torch.equal(w, wc) and torch.equal(b, bc) and torch.equal(y, yc)


def test_invalid_reduction_rejected():
    x, w, b, y = make_inputs(seed=17)
    with pytest.raises(ValueError):
        reference(x, w, y, b, reduction="batchmean")
