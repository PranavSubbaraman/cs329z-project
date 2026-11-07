"""Math checks for benchmark/kernels/ce/reference.py (pure torch, no liger, no triton).

Run: .venv/bin/python -m pytest tests/test_reference_ce.py -q
"""

import importlib.util
import itertools
import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

_REF_PATH = pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "kernels" / "ce" / "reference.py"
_spec = importlib.util.spec_from_file_location("ce_reference", _REF_PATH)
_mod = importlib.util.module_from_spec(_spec)
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep benchmark/kernels/ce free of __pycache__
try:
    _spec.loader.exec_module(_mod)
finally:
    sys.dont_write_bytecode = _dwb
reference = _mod.reference

D = torch.float64
REDUCTIONS = ["mean", "sum", "none"]


def _make(bt, v, seed, ignore_index=-100, n_ignored=None, scale=2.0, dtype=D):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(bt, v, generator=g, dtype=torch.float64) * scale).to(dtype)
    t = torch.randint(0, v, (bt,), generator=g)
    if 0 <= ignore_index < v:
        # the class id `ignore_index` appears only on the rows we ignore on purpose
        t = torch.where(t == ignore_index, (t + 1) % v, t)
    if n_ignored is None:
        n_ignored = bt // 3
    if n_ignored:
        idx = torch.randperm(bt, generator=g)[:n_ignored]
        t[idx] = ignore_index
    return x, t


def _loss_and_grad(fn, x, upstream=None, **kw):
    xx = x.detach().clone().requires_grad_(True)
    out = fn(xx, **kw)
    if upstream is None:
        upstream = torch.ones_like(out)
    (g,) = torch.autograd.grad(out, xx, grad_outputs=upstream)
    return out.detach(), g


def _torch_ce(x, target, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    return F.cross_entropy(x, target, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)


def _torch_ce_with_z(x, target, ignore_index=-100, reduction="mean", label_smoothing=0.0, lse_square_scale=0.0):
    """Independent formulation: torch's cross_entropy plus s * logsumexp^2 on counted rows."""
    ce = _torch_ce(x, target, ignore_index, reduction, label_smoothing)
    counted = target != ignore_index
    z = torch.where(counted, lse_square_scale * torch.logsumexp(x, -1) ** 2, torch.zeros((), dtype=x.dtype))
    if reduction == "none":
        return ce + z
    if reduction == "sum":
        return ce + z.sum()
    return ce + z.sum() / counted.sum()


# ----------------------------------------------------------------------------------------------
# Agreement with torch built-ins
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("ignore_index", [-100, 3, -7])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("n_ignored", [0, 4])
def test_matches_torch_cross_entropy(reduction, ignore_index, label_smoothing, n_ignored):
    x, t = _make(13, 11, seed=1, ignore_index=ignore_index, n_ignored=n_ignored)
    up = torch.rand(13, dtype=D) - 0.3 if reduction == "none" else torch.tensor(1.7, dtype=D)
    kw = dict(target=t, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    lr, gr = _loss_and_grad(reference, x, up, **kw)
    lt, gt = _loss_and_grad(_torch_ce, x, up, **kw)
    torch.testing.assert_close(lr, lt, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gr, gt, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.2])
@pytest.mark.parametrize("lse_square_scale", [1e-4, 1e-2, 0.3])
def test_z_term_matches_independent_formula(reduction, label_smoothing, lse_square_scale):
    x, t = _make(12, 10, seed=3, scale=3.0)
    up = torch.randn(12, dtype=D) if reduction == "none" else torch.tensor(0.6, dtype=D)
    kw = dict(target=t, reduction=reduction, label_smoothing=label_smoothing, lse_square_scale=lse_square_scale)
    lr, gr = _loss_and_grad(reference, x, up, **kw)
    lt, gt = _loss_and_grad(_torch_ce_with_z, x, up, **kw)
    torch.testing.assert_close(lr, lt, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gr, gt, rtol=1e-12, atol=1e-12)


def test_matches_nll_of_log_softmax():
    x, t = _make(9, 6, seed=2, n_ignored=3)
    a = reference(x, t, reduction="none")
    b = F.nll_loss(torch.log_softmax(x, -1), t, ignore_index=-100, reduction="none")
    torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


# ----------------------------------------------------------------------------------------------
# gradcheck / gradgradcheck (float64, random points; the op is smooth in `input`, no kinks)
# ----------------------------------------------------------------------------------------------
_GC_CASES = list(itertools.product(REDUCTIONS, [0.0, 0.3, 1.0], [0.0, 0.05], [-100, 2]))


@pytest.mark.parametrize("reduction,label_smoothing,lse_square_scale,ignore_index", _GC_CASES)
def test_gradcheck(reduction, label_smoothing, lse_square_scale, ignore_index):
    x, t = _make(6, 7, seed=4, ignore_index=ignore_index, n_ignored=2)
    x = x.requires_grad_(True)

    def f(xx):
        return reference(
            xx,
            t,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            lse_square_scale=lse_square_scale,
        )

    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-7, rtol=1e-6)
    assert torch.autograd.gradgradcheck(f, (x,), eps=1e-6, atol=1e-6, rtol=1e-5)


# ----------------------------------------------------------------------------------------------
# Closed-form gradients
# ----------------------------------------------------------------------------------------------
def _closed_form_grad(x, t, ignore_index, reduction, ls, s, upstream):
    bt, v = x.shape
    counted = (t != ignore_index).to(D)
    safe_t = torch.where(t != ignore_index, t, torch.zeros_like(t))
    p = torch.softmax(x, -1)
    lse = torch.logsumexp(x, -1)
    onehot = F.one_hot(safe_t, v).to(D)
    # d/dx_j [ (1-ls) nll + ls * mean_c(-log p_c) + s lse^2 ] = p_j - (1-ls) 1[j=y] - ls/V + 2 s lse p_j
    g = p - (1 - ls) * onehot - ls / v + 2 * s * lse[:, None] * p
    g = g * counted[:, None]
    if reduction == "mean":
        g = g / counted.sum().clamp(min=1)
    return g * (upstream[:, None] if reduction == "none" else upstream)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("lse_square_scale", [0.0, 0.01])
@pytest.mark.parametrize("ignore_index", [-100, 2])
def test_closed_form_gradient(reduction, label_smoothing, lse_square_scale, ignore_index):
    x, t = _make(10, 9, seed=5, ignore_index=ignore_index, n_ignored=3, scale=3.0)
    up = torch.rand(10, dtype=D) + 0.5 if reduction == "none" else torch.tensor(0.8, dtype=D)
    _, g = _loss_and_grad(
        reference,
        x,
        up,
        target=t,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        lse_square_scale=lse_square_scale,
    )
    expect = _closed_form_grad(x, t, ignore_index, reduction, label_smoothing, lse_square_scale, up)
    torch.testing.assert_close(g, expect, rtol=1e-11, atol=1e-13)


def test_plain_mean_gradient_is_softmax_minus_onehot_over_n():
    x, t = _make(8, 5, seed=6, n_ignored=0)
    _, g = _loss_and_grad(reference, x, target=t)
    expect = (torch.softmax(x, -1) - F.one_hot(t, 5).to(D)) / 8
    torch.testing.assert_close(g, expect, rtol=1e-13, atol=1e-15)


# ----------------------------------------------------------------------------------------------
# Op-specific math facts
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_row_shift_invariance_without_z_term(reduction):
    x, t = _make(7, 6, seed=7)
    shift = torch.randn(7, 1, dtype=D) * 10
    a = reference(x, t, reduction=reduction, label_smoothing=0.3)
    b = reference(x + shift, t, reduction=reduction, label_smoothing=0.3)
    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


def test_row_shift_changes_z_term_exactly():
    s = 0.02
    x, t = _make(7, 6, seed=8, n_ignored=0)
    c = torch.randn(7, dtype=D)
    a = reference(x, t, reduction="none", lse_square_scale=s)
    b = reference(x + c[:, None], t, reduction="none", lse_square_scale=s)
    lse = torch.logsumexp(x, -1)
    torch.testing.assert_close(b - a, s * ((lse + c) ** 2 - lse**2), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.5, 1.0])
def test_uniform_logits_give_log_v(label_smoothing):
    v = 13
    x = torch.full((5, v), 0.37, dtype=D)
    t = torch.tensor([0, 4, 12, 7, 1])
    out = reference(x, t, label_smoothing=label_smoothing)
    torch.testing.assert_close(out, torch.tensor(math.log(v), dtype=D), rtol=1e-14, atol=1e-14)


def test_reduction_consistency():
    x, t = _make(11, 8, seed=9)
    kw = dict(label_smoothing=0.15, lse_square_scale=0.01)
    none = reference(x, t, reduction="none", **kw)
    total = reference(x, t, reduction="sum", **kw)
    mean = reference(x, t, reduction="mean", **kw)
    counted = t != -100
    torch.testing.assert_close(none.sum(), total, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(mean, total / counted.sum(), rtol=1e-13, atol=1e-13)
    assert torch.all(none[~counted] == 0)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ignored_rows_do_not_matter(reduction):
    x, t = _make(9, 7, seed=10, n_ignored=4)
    ign = t == -100
    x2 = x.clone()
    x2[ign] = torch.randn(int(ign.sum()), 7, dtype=D) * 50
    up = torch.ones(9, dtype=D) if reduction == "none" else None
    kw = dict(target=t, reduction=reduction, label_smoothing=0.2, lse_square_scale=0.01)
    l1, g1 = _loss_and_grad(reference, x, up, **kw)
    l2, g2 = _loss_and_grad(reference, x2, up, **kw)
    torch.testing.assert_close(l1, l2, rtol=1e-13, atol=1e-13)
    assert torch.all(g1[ign] == 0) and torch.all(g2[ign] == 0)
    torch.testing.assert_close(g1[~ign], g2[~ign], rtol=1e-13, atol=1e-15)


def test_ignore_index_inside_class_range():
    """A target equal to ignore_index is ignored even when ignore_index is a valid class id."""
    x, t = _make(10, 6, seed=11, ignore_index=2, n_ignored=3)
    counted = t != 2
    for red in ("sum", "mean"):
        a = reference(x, t, ignore_index=2, reduction=red, label_smoothing=0.1)
        b = reference(x[counted], t[counted], ignore_index=-100, reduction=red, label_smoothing=0.1)
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_row_permutation_invariance(reduction):
    x, t = _make(10, 5, seed=12)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    kw = dict(reduction=reduction, label_smoothing=0.1, lse_square_scale=0.01)
    torch.testing.assert_close(reference(x, t, **kw), reference(x[perm], t[perm], **kw), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_all_rows_ignored(reduction):
    x, t = _make(4, 5, seed=13, n_ignored=4)
    assert torch.all(t == -100)
    up = torch.ones(4, dtype=D) if reduction == "none" else None
    out, g = _loss_and_grad(reference, x, up, target=t, reduction=reduction, label_smoothing=0.3, lse_square_scale=0.1)
    assert torch.all(out == 0) and torch.isfinite(out).all()
    assert torch.all(g == 0)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.3, 1.0])
def test_nonnegative(label_smoothing):
    x, t = _make(20, 9, seed=14, scale=5.0)
    out = reference(x, t, reduction="none", label_smoothing=label_smoothing, lse_square_scale=0.01)
    assert torch.all(out >= 0)


def test_full_smoothing_ignores_target_class():
    """With label_smoothing = 1 the per-row loss is mean_c(-log p_c), independent of the target class."""
    x, t = _make(6, 8, seed=15, n_ignored=0)
    t2 = (t + 3) % 8
    a = reference(x, t, reduction="none", label_smoothing=1.0)
    b = reference(x, t2, reduction="none", label_smoothing=1.0)
    torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(a, -torch.log_softmax(x, -1).mean(-1), rtol=1e-13, atol=1e-13)


def test_gradient_rows_sum_to_zero_without_z_term():
    x, t = _make(9, 7, seed=16)
    _, g = _loss_and_grad(reference, x, target=t, reduction="sum", label_smoothing=0.35)
    torch.testing.assert_close(g.sum(-1), torch.zeros(9, dtype=D), rtol=0, atol=1e-13)


def test_gradient_row_sum_with_z_term_is_2_s_lse():
    s = 0.05
    x, t = _make(9, 7, seed=17, n_ignored=0)
    _, g = _loss_and_grad(reference, x, target=t, reduction="sum", lse_square_scale=s)
    torch.testing.assert_close(g.sum(-1), 2 * s * torch.logsumexp(x, -1), rtol=1e-12, atol=1e-13)


def test_confident_true_class_gradient_has_no_cancellation_in_float64():
    """When softmax(x_y) -> 1 the true-class gradient (p_y - 1)/N must stay accurate; compare with -sum_{j!=y} p_j/N."""
    x, t = _make(6, 50, seed=18, n_ignored=0)
    x[torch.arange(6), t] = x.max(-1).values + 12.0
    _, g = _loss_and_grad(reference, x, target=t)
    p = torch.softmax(x, -1)
    others = p.clone()
    others[torch.arange(6), t] = 0
    expect = -others.sum(-1) / 6
    torch.testing.assert_close(g[torch.arange(6), t], expect, rtol=1e-9, atol=0)


# ----------------------------------------------------------------------------------------------
# Contract details: dtypes, constants, layouts, no mutation of arguments
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_output_dtype_follows_input(dtype):
    x, t = _make(5, 4, seed=19, dtype=dtype)
    for red in REDUCTIONS:
        out = reference(x, t, reduction=red, label_smoothing=0.1, lse_square_scale=0.01)
        assert out.dtype == dtype
        assert out.shape == ((5,) if red == "none" else ())


def test_target_gets_no_gradient_and_is_constant():
    x, t = _make(6, 5, seed=20)
    x.requires_grad_(True)
    out = reference(x, t, label_smoothing=0.2)
    out.backward()
    assert x.grad is not None
    assert t.grad is None and not t.requires_grad


def test_noncontiguous_input_same_result():
    x, t = _make(6, 5, seed=21)
    big = torch.randn(6, 9, dtype=D)
    big[:, 2:7] = x
    view = big[:, 2:7]
    tr = x.t().contiguous().t()
    assert not view.is_contiguous() and not tr.is_contiguous()
    kw = dict(reduction="none", label_smoothing=0.1, lse_square_scale=0.01)
    ref = reference(x, t, **kw)
    torch.testing.assert_close(reference(view, t, **kw), ref, rtol=0, atol=0)
    torch.testing.assert_close(reference(tr, t, **kw), ref, rtol=1e-15, atol=1e-15)


def test_reference_does_not_modify_arguments():
    x, t = _make(6, 5, seed=22, n_ignored=2)
    xs, ts = x.clone(), t.clone()
    for rg in (False, True):
        xx = x.clone().requires_grad_(rg)
        out = reference(xx, t, reduction="sum", label_smoothing=0.2, lse_square_scale=0.01)
        if rg:
            out.backward()
        assert torch.equal(xx.detach(), xs) and torch.equal(t, ts)


def test_rejects_unknown_reduction():
    x, t = _make(3, 4, seed=23)
    with pytest.raises(ValueError):
        reference(x, t, reduction="batchmean")
