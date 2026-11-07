"""HARNESS-SIDE math checks for benchmark/kernels/tvd/reference.py (pure torch: no liger, no triton).

Run:  .venv/bin/python -m pytest tests/test_reference_tvd.py -q

Covers: gradcheck in float64 away from the |.| kink, closed-form gradients for every reduction (with and without
ignored rows, with scalar and (BT, V) upstream gradients), exactly-zero gradient at exact ties p == q (full, partial
and zero-padding ties) plus central finite differences at a tie, ignore_index semantics, reduction identities,
torch built-in equivalents (F.l1_loss, vector_norm), total-variation facts, constness of q, dtype handling and
out-of-scope errors.
"""
import importlib.util
import pathlib

import pytest
import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tvd_reference", _ROOT / "benchmark/kernels/tvd/reference.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
reference = _mod.reference

REDUCTIONS = ["none", "sum", "mean", "batchmean"]
IGNORE_INDICES = [-100, 0, 1]
DT = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _inputs(BT, V, seed, normalized=True):
    g = _gen(seed)
    if normalized:
        p = torch.randn(BT, V, generator=g, dtype=DT).softmax(-1)
        q = torch.randn(BT, V, generator=g, dtype=DT).softmax(-1)
    else:
        p = torch.randn(BT, V, generator=g, dtype=DT)
        q = torch.randn(BT, V, generator=g, dtype=DT).softmax(-1)
    return p, q


def _labels(BT, V, ignore_index, seed, n_ignored=None):
    """Random labels with some rows set to ignore_index and at least one row kept."""
    g = _gen(seed + 1000)
    lab = torch.randint(0, V, (BT,), generator=g, dtype=torch.long)
    lab[lab == ignore_index] = ignore_index + 2  # rows are ignored only where we choose
    n_ignored = max(1, BT // 3) if n_ignored is None else n_ignored
    idx = torch.randperm(BT, generator=g)[:n_ignored]
    lab[idx] = ignore_index
    assert (lab != ignore_index).any()
    return lab


def _mask(BT, labels, ignore_index):
    if labels is None:
        return torch.ones(BT, 1, dtype=DT)
    return (labels != ignore_index).to(DT).unsqueeze(1)


def _scale(reduction, labels, ignore_index, BT, V):
    n = BT if labels is None else int((labels != ignore_index).sum())
    return {"none": 1.0, "sum": 1.0, "mean": 1.0 / (n * V), "batchmean": 1.0 / n}[reduction]


def _upstream(reduction, BT, V, seed):
    if reduction == "none":
        return torch.rand(BT, V, generator=_gen(seed + 7), dtype=DT) + 0.5  # in [0.5, 1.5)
    return torch.tensor(1.7, dtype=DT)


def _grad(p, q, labels, reduction, ignore_index, upstream):
    x = p.detach().clone().requires_grad_(True)
    out = reference(x, q, shift_labels=labels, reduction=reduction, ignore_index=ignore_index)
    (g,) = torch.autograd.grad(out, x, grad_outputs=upstream)
    return out.detach(), g


def _closed_form(p, q, labels, reduction, ignore_index, upstream):
    BT, V = p.shape
    s = _scale(reduction, labels, ignore_index, BT, V)
    return 0.5 * torch.sign(p - q) * _mask(BT, labels, ignore_index) * s * upstream


def _away_from_kink(p, q, margin=0.05):
    """Move p so that |p - q| >= margin everywhere (keeps the sign pattern random)."""
    d = p - q
    return q + torch.where(d >= 0, 1.0, -1.0) * (d.abs() + margin)


# --------------------------------------------------------------------------------------------------------------
# gradcheck and closed-form derivatives
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("ignore_index", [None] + IGNORE_INDICES)
@pytest.mark.parametrize("seed", [0, 1])
def test_gradcheck_away_from_kink(reduction, ignore_index, seed):
    BT, V = 5, 7
    p, q = _inputs(BT, V, seed, normalized=bool(seed % 2))
    p = _away_from_kink(p, q)
    labels = None if ignore_index is None else _labels(BT, V, ignore_index, seed)
    ii = -100 if ignore_index is None else ignore_index
    x = p.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda t: reference(t, q, shift_labels=labels, reduction=reduction, ignore_index=ii),
        (x,),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-5,
    )


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("ignore_index", [None] + IGNORE_INDICES)
@pytest.mark.parametrize("normalized", [True, False])
def test_closed_form_gradient(reduction, ignore_index, normalized):
    BT, V = 9, 13
    p, q = _inputs(BT, V, 3, normalized=normalized)
    labels = None if ignore_index is None else _labels(BT, V, ignore_index, 3)
    ii = -100 if ignore_index is None else ignore_index
    up = _upstream(reduction, BT, V, 3)
    _, g = _grad(p, q, labels, reduction, ii, up)
    torch.testing.assert_close(g, _closed_form(p, q, labels, reduction, ii, up), rtol=1e-14, atol=0)


# --------------------------------------------------------------------------------------------------------------
# exact ties p == q: d/dp 0.5|p - q| = 0.5 sign(p - q) with sign(0) = 0
# --------------------------------------------------------------------------------------------------------------


def _tied_inputs(BT, V, pattern, seed):
    p, q = _inputs(BT, V, seed)
    q = q.clone()
    tied = torch.zeros(BT, V, dtype=torch.bool)
    if pattern == "full":  # identical distributions
        q = p.clone()
        tied[:] = True
    elif pattern == "partial":  # trailing half of the columns tied
        q[:, V // 2 :] = p[:, V // 2 :]
        tied[:, V // 2 :] = True
    elif pattern == "zeros":  # trailing columns exactly 0.0 in both, rows renormalized
        k = 3
        p = p.clone()
        p[:, V - k :] = 0.0
        q[:, V - k :] = 0.0
        p = p / p.sum(-1, keepdim=True)
        q = q / q.sum(-1, keepdim=True)
        tied[:, V - k :] = True
    elif pattern == "scattered":  # random positions tied
        sel = torch.rand(BT, V, generator=_gen(seed + 5)) < 0.3
        q[sel] = p[sel]
        tied = sel
    else:
        raise AssertionError(pattern)
    assert torch.equal(p[tied], q[tied]) and not (p[~tied] == q[~tied]).any()
    return p, q, tied


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("ignore_index", [None] + IGNORE_INDICES)
@pytest.mark.parametrize("pattern", ["full", "partial", "zeros", "scattered"])
def test_zero_gradient_at_exact_ties(reduction, ignore_index, pattern):
    BT, V = 6, 10
    p, q, tied = _tied_inputs(BT, V, pattern, seed=11)
    labels = None if ignore_index is None else _labels(BT, V, ignore_index, 11, n_ignored=2)
    ii = -100 if ignore_index is None else ignore_index
    up = _upstream(reduction, BT, V, 11)
    out, g = _grad(p, q, labels, reduction, ii, up)
    # exactly zero (not just small) at every tied element
    assert (g[tied] == 0).all(), g[tied]
    # untied elements of kept rows have the full magnitude 0.5 * scale * |upstream|
    keep = _mask(BT, labels, ii).bool().expand(BT, V)
    s = _scale(reduction, labels, ii, BT, V)
    mag = (0.5 * s * up.abs()).expand(BT, V)
    sel = ~tied & keep
    torch.testing.assert_close(g[sel].abs(), mag[sel], rtol=1e-14, atol=0)
    assert (g[~keep] == 0).all()
    torch.testing.assert_close(g, _closed_form(p, q, labels, reduction, ii, up), rtol=1e-14, atol=0)
    if pattern == "full":
        assert float(out.abs().max()) == 0.0  # identical inputs: zero loss


def _scalar_objective(p, q, labels, reduction, ignore_index, up):
    out = reference(p, q, shift_labels=labels, reduction=reduction, ignore_index=ignore_index)
    return (out * up).sum()


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("ignore_index", [None] + IGNORE_INDICES)
@pytest.mark.parametrize("h", [1e-3, 1e-6])
def test_central_difference_at_exact_tie(reduction, ignore_index, h):
    BT, V = 4, 8
    p, q, tied = _tied_inputs(BT, V, "partial", seed=21)
    labels = None if ignore_index is None else _labels(BT, V, ignore_index, 21, n_ignored=1)
    ii = -100 if ignore_index is None else ignore_index
    keep_rows = torch.arange(BT) if labels is None else torch.nonzero(labels != ii).flatten()
    i = int(keep_rows[0])
    up = _upstream(reduction, BT, V, 21)
    _, g = _grad(p, q, labels, reduction, ii, up)
    s = _scale(reduction, labels, ii, BT, V)
    for j in range(V):
        e = torch.zeros(BT, V, dtype=DT)
        e[i, j] = h
        f_plus = _scalar_objective(p + e, q, labels, reduction, ii, up)
        f_minus = _scalar_objective(p - e, q, labels, reduction, ii, up)
        f0 = _scalar_objective(p, q, labels, reduction, ii, up)
        cd = float((f_plus - f_minus) / (2 * h))
        unit = 0.5 * s * float(up if up.dim() == 0 else up[i, j])
        if tied[i, j]:
            assert float(g[i, j]) == 0.0
            assert abs(cd) <= 1e-9 * unit, (j, cd)
            # one-sided slopes are +-0.5*scale*upstream: the kink is real, and the central difference is its midpoint
            assert abs(float((f_plus - f0) / h) - unit) <= 1e-6 * unit
            assert abs(float((f0 - f_minus) / h) + unit) <= 1e-6 * unit
        else:
            assert abs(cd - float(g[i, j])) <= 1e-6 * unit, (j, cd, float(g[i, j]))


@pytest.mark.parametrize("pdt, qdt", [(torch.float32, torch.float32), (torch.bfloat16, torch.float32),
                                      (torch.float32, torch.bfloat16)])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_float64_cast_preserves_ties(pdt, qdt, reduction):
    """The oracle casts native inputs to float64: ties in the native dtypes stay exactly the ties after the cast."""
    g = _gen(25)
    p = torch.randn(6, 33, generator=g).softmax(-1).to(pdt)
    q = torch.randn(6, 33, generator=g).softmax(-1).to(qdt)
    sel = torch.rand(6, 33, generator=g) < 0.4
    # tie positions: copy p into q (exact when q's dtype can hold p's values, else copy q into p)
    if qdt == torch.bfloat16 and pdt != torch.bfloat16:
        p = torch.where(sel, q.to(pdt), p)
    else:
        q = torch.where(sel, p.to(qdt), q)
    tied_native = p.to(torch.float32) == q.to(torch.float32)
    p64, q64 = p.double(), q.double()
    assert torch.equal(tied_native, p64 == q64) and torch.equal(tied_native, sel | tied_native)
    up = _upstream(reduction, 6, 33, 25)
    _, g64 = _grad(p64, q64, None, reduction, -100, up)
    assert (g64[tied_native] == 0).all() and (g64[~tied_native] != 0).all()


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_tie_gradient_matches_torch_builtin_l1(reduction):
    """F.l1_loss's autograd also uses sign(0) = 0; for unmasked inputs the reference is 0.5 * l1_loss."""
    BT, V = 5, 9
    p, q, tied = _tied_inputs(BT, V, "scattered", seed=31)
    up = _upstream(reduction, BT, V, 31)
    _, g = _grad(p, q, None, reduction, -100, up)
    x = p.clone().requires_grad_(True)
    if reduction == "batchmean":
        out = 0.5 * F.l1_loss(x, q, reduction="sum") / BT
    else:
        out = 0.5 * F.l1_loss(x, q, reduction=reduction)
    (g2,) = torch.autograd.grad(out, x, grad_outputs=up)
    torch.testing.assert_close(g, g2, rtol=1e-14, atol=0)
    assert (g2[tied] == 0).all()


# --------------------------------------------------------------------------------------------------------------
# reductions, ignore_index, built-ins
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ignore_index", [None] + IGNORE_INDICES)
@pytest.mark.parametrize("normalized", [True, False])
def test_reduction_identities(ignore_index, normalized):
    BT, V = 7, 11
    p, q = _inputs(BT, V, 41, normalized=normalized)
    labels = None if ignore_index is None else _labels(BT, V, ignore_index, 41)
    ii = -100 if ignore_index is None else ignore_index
    n = BT if labels is None else int((labels != ii).sum())
    r = {red: reference(p, q, shift_labels=labels, reduction=red, ignore_index=ii) for red in REDUCTIONS}
    assert r["none"].shape == (BT, V) and all(r[k].shape == () for k in ("sum", "mean", "batchmean"))
    torch.testing.assert_close(r["sum"], r["none"].sum(), rtol=1e-14, atol=0)
    torch.testing.assert_close(r["mean"], r["sum"] / (n * V), rtol=1e-14, atol=0)
    torch.testing.assert_close(r["batchmean"], r["sum"] / n, rtol=1e-14, atol=0)
    torch.testing.assert_close(r["none"], 0.5 * (p - q).abs() * _mask(BT, labels, ii), rtol=1e-14, atol=0)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_default_arguments(reduction):
    p, q = _inputs(3, 5, 43)
    torch.testing.assert_close(reference(p, q), reference(p, q, None, "batchmean", -100), rtol=0, atol=0)
    lab = torch.tensor([-100, 2, 3])
    torch.testing.assert_close(
        reference(p, q, lab, reduction), reference(p[1:], q[1:], None, reduction) if reduction != "none" else
        torch.cat([torch.zeros(1, 5, dtype=DT), reference(p[1:], q[1:], None, "none")]), rtol=1e-14, atol=0)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("ignore_index", IGNORE_INDICES)
def test_ignored_rows(reduction, ignore_index):
    BT, V = 8, 6
    p, q = _inputs(BT, V, 51)
    labels = _labels(BT, V, ignore_index, 51, n_ignored=3)
    ign = labels == ignore_index
    up = _upstream(reduction, BT, V, 51)
    out, g = _grad(p, q, labels, reduction, ignore_index, up)
    # ignored rows get zero gradient and, for "none", zero loss
    assert (g[ign] == 0).all()
    if reduction == "none":
        assert (out[ign] == 0).all()
    # the loss does not depend on the values in ignored rows
    p2, q2 = p.clone(), q.clone()
    p2[ign] = 123.0
    q2[ign] = -7.0
    out2, g2 = _grad(p2, q2, labels, reduction, ignore_index, up)
    torch.testing.assert_close(out2, out, rtol=0, atol=0)
    torch.testing.assert_close(g2, g, rtol=0, atol=0)
    # equals the op on the kept rows only (N = number of kept rows)
    keep = ~ign
    sub = reference(p[keep], q[keep], reduction=reduction)
    torch.testing.assert_close(out[keep] if reduction == "none" else out, sub, rtol=1e-14, atol=0)
    # label values other than ignore_index do not matter
    lab3 = labels.clone()
    lab3[keep] = ignore_index + 5
    torch.testing.assert_close(reference(p, q, lab3, reduction, ignore_index), out, rtol=0, atol=0)
    # labels present but none equal to ignore_index == no labels
    lab4 = torch.full((BT,), ignore_index + 1, dtype=torch.long)
    torch.testing.assert_close(reference(p, q, lab4, reduction, ignore_index), reference(p, q, None, reduction),
                               rtol=0, atol=0)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_matches_torch_builtins(reduction):
    BT, V = 6, 17
    p, q = _inputs(BT, V, 61, normalized=False)
    r = reference(p, q, reduction=reduction)
    if reduction == "batchmean":
        b = 0.5 * torch.linalg.vector_norm(p - q, ord=1, dim=-1).mean()
    else:
        b = 0.5 * F.l1_loss(p, q, reduction=reduction)
    torch.testing.assert_close(r, b, rtol=1e-14, atol=0)


def test_total_variation_facts():
    BT, V = 16, 12
    p, q = _inputs(BT, V, 71)
    _, r = _inputs(BT, V, 72)
    tv = reference(p, q, reduction="none").sum(-1)
    assert ((tv >= 0) & (tv <= 1)).all()
    torch.testing.assert_close(tv, reference(q, p, reduction="none").sum(-1), rtol=0, atol=0)  # symmetric
    torch.testing.assert_close(tv, (p - q).clamp(min=0).sum(-1), rtol=1e-12, atol=1e-15)  # sum of positive part
    event = p > q  # TV = max over events A of |P(A) - Q(A)|, attained at A = {p > q}
    torch.testing.assert_close(tv, ((p - q) * event).sum(-1), rtol=1e-12, atol=1e-15)
    for _ in range(50):
        a = torch.rand(BT, V, generator=_gen(_ + 100)) < 0.5
        assert (((p - q) * a).sum(-1).abs() <= tv + 1e-12).all()
    tv_pr = reference(p, r, reduction="none").sum(-1)
    tv_qr = reference(q, r, reduction="none").sum(-1)
    assert (tv_pr <= tv + tv_qr + 1e-12).all()  # triangle inequality
    a = torch.zeros(2, 4, dtype=DT)
    b = torch.zeros(2, 4, dtype=DT)
    a[:, :2] = 0.5
    b[:, 2:] = 0.5
    torch.testing.assert_close(reference(a, b, reduction="batchmean"), torch.tensor(1.0, dtype=DT))  # disjoint
    assert float(reference(p, p.clone(), reduction="sum")) == 0.0


# --------------------------------------------------------------------------------------------------------------
# contract details
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_q_is_constant(reduction):
    p, q = _inputs(4, 6, 81)
    x = p.clone().requires_grad_(True)
    t = q.clone().requires_grad_(True)
    out = reference(x, t, reduction=reduction)
    out.backward(torch.ones_like(out))
    assert t.grad is None and x.grad is not None


def test_dtypes_follow_inputs():
    p, q = _inputs(3, 5, 91)
    assert reference(p, q, reduction="sum").dtype == torch.float64
    assert reference(p.float(), q.float(), reduction="none").dtype == torch.float32
    assert reference(p.bfloat16(), q.float(), reduction="mean").dtype == torch.float32  # promotion
    x = p.clone().requires_grad_(True)
    reference(x, q, reduction="batchmean").backward()
    assert x.grad.dtype == torch.float64


def test_noncontiguous_inputs():
    p, q = _inputs(5, 9, 101)
    pt, qt = p.t().contiguous().t(), q.t().contiguous().t()
    assert not pt.is_contiguous()
    lab = torch.tensor([0, -100, 2, 3, -100])
    for red in REDUCTIONS:
        torch.testing.assert_close(reference(pt, qt, lab, red), reference(p, q, lab, red), rtol=0, atol=0)


def test_all_rows_ignored_and_invalid_reduction():
    p, q = _inputs(3, 4, 111)
    lab = torch.full((3,), -100, dtype=torch.long)
    x = p.clone().requires_grad_(True)
    out = reference(x, q, lab, "sum")
    out.backward()
    assert float(out.detach()) == 0.0 and (x.grad == 0).all()
    assert (reference(p, q, lab, "none") == 0).all()
    for red in ("mean", "batchmean"):
        with pytest.raises(ValueError):
            reference(p, q, lab, red)
    with pytest.raises(ValueError):
        reference(p, q, reduction="avg")
