"""Math checks for benchmark/kernels/fljsd/reference.py (pure torch, no liger).

Run:  .venv/bin/python -m pytest tests/test_reference_fljsd.py -q
"""

import importlib.util
import math
import os

import pytest
import torch
import torch.nn.functional as F

_REF_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmark", "kernels", "fljsd", "reference.py")
_spec = importlib.util.spec_from_file_location("fljsd_reference", _REF_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
reference = _mod.reference

F64 = torch.float64


def _inputs(BT=5, Hs=7, Ht=9, V=11, scale=1.0, seed=0, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(BT, Hs, generator=g, dtype=dtype)
    ws = torch.randn(V, Hs, generator=g, dtype=dtype) * (scale / Hs**0.5)
    xt = torch.randn(BT, Ht, generator=g, dtype=dtype)
    wt = torch.randn(V, Ht, generator=g, dtype=dtype) * (scale / Ht**0.5)
    return xs, ws, xt, wt


def _probs(xs, ws, xt, wt, T):
    log_q = torch.log_softmax(xs @ ws.t() / T, -1)
    log_p = torch.log_softmax(xt @ wt.t() / T, -1)
    return log_q, log_p


def _grads(loss, *ts):
    return torch.autograd.grad(loss, ts)


BETAS = [0.0, 0.1, 0.5, 0.73, 1.0]


# ----------------------------------------------------------------------------------------------- gradcheck
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("T", [1.0, 1.7, 0.6])
@pytest.mark.parametrize("labels", ["none", "partial"])
def test_gradcheck_float64(beta, T, labels):
    xs, ws, xt, wt = _inputs(BT=4, Hs=3, Ht=5, V=6, scale=2.0, seed=int(10 * beta + 3 * T))
    lab = None
    if labels == "partial":
        lab = torch.tensor([2, -100, 0, 5])
    xs.requires_grad_(True)
    ws.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, b: reference(a, b, xt, wt, shift_labels=lab, jsd_beta=beta, temperature=T), (xs, ws)
    )
    assert torch.autograd.gradgradcheck(
        lambda a, b: reference(a, b, xt, wt, shift_labels=lab, jsd_beta=beta, temperature=T), (xs, ws)
    )


# ------------------------------------------------------------------------------- closed-form derivatives
def _closed_form_G(xs, ws, xt, wt, beta, T, keep):
    """dL/dz for z = X_s W_s^T (before the 1/T scaling), derived by hand."""
    log_q, log_p = _probs(xs, ws, xt, wt, T)
    q, p = log_q.exp(), log_p.exp()
    N = max(int(keep.sum()), 1)
    if beta == 0.0:
        G = q - p
    elif beta == 1.0:
        kl = (q * (log_q - log_p)).sum(-1, keepdim=True)
        G = q * (log_q - log_p - kl)
    else:
        log_m = torch.log(beta * p + (1 - beta) * q)
        a = q * (log_q - log_m)
        G = (1 - beta) * (a - q * a.sum(-1, keepdim=True))
    return G * keep[:, None].to(G.dtype) / (N * T)


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("T", [1.0, 2.5])
@pytest.mark.parametrize("labels", ["none", "partial", "all"])
def test_closed_form_gradients(beta, T, labels):
    xs, ws, xt, wt = _inputs(BT=6, Hs=4, Ht=8, V=13, scale=3.0, seed=7)
    lab = None
    keep = torch.ones(6, dtype=torch.bool)
    if labels != "none":
        lab = torch.tensor([1, 42, 3, 42, 0, 12]) if labels == "partial" else torch.full((6,), 42)
        keep = lab != 42
    xs.requires_grad_(True)
    ws.requires_grad_(True)
    loss = reference(xs, ws, xt, wt, shift_labels=lab, jsd_beta=beta, ignore_index=42, temperature=T)
    gx, gw = _grads(loss, xs, ws)
    with torch.no_grad():
        G = _closed_form_G(xs, ws, xt, wt, beta, T, keep)
    torch.testing.assert_close(gx, G @ ws.detach(), rtol=1e-10, atol=1e-13)
    torch.testing.assert_close(gw, G.t() @ xs.detach(), rtol=1e-10, atol=1e-13)


# ---------------------------------------------------------------------------- comparison with built-ins
@pytest.mark.parametrize("T", [1.0, 1.3])
def test_endpoints_match_torch_kl_div(T):
    xs, ws, xt, wt = _inputs(BT=7, V=17, scale=4.0, seed=1)
    log_q, log_p = _probs(xs, ws, xt, wt, T)
    fkl = F.kl_div(log_q, log_p, reduction="sum", log_target=True) / 7  # KL(P || Q)
    rkl = F.kl_div(log_p, log_q, reduction="sum", log_target=True) / 7  # KL(Q || P)
    torch.testing.assert_close(reference(xs, ws, xt, wt, jsd_beta=0.0, temperature=T), fkl, rtol=1e-12, atol=0)
    torch.testing.assert_close(reference(xs, ws, xt, wt, jsd_beta=1.0, temperature=T), rkl, rtol=1e-12, atol=0)


@pytest.mark.parametrize("beta", [0.1, 0.5, 0.73, 0.99])
def test_interior_matches_kl_div_to_mixture(beta):
    xs, ws, xt, wt = _inputs(BT=7, V=17, scale=4.0, seed=2)
    log_q, log_p = _probs(xs, ws, xt, wt, 1.0)
    log_m = torch.log(beta * log_p.exp() + (1 - beta) * log_q.exp())
    expect = (
        beta * F.kl_div(log_m, log_p, reduction="sum", log_target=True)
        + (1 - beta) * F.kl_div(log_m, log_q, reduction="sum", log_target=True)
    ) / 7
    torch.testing.assert_close(reference(xs, ws, xt, wt, jsd_beta=beta), expect, rtol=1e-12, atol=1e-15)


def test_logits_match_linear():
    xs, ws, xt, wt = _inputs(seed=3)
    log_q = torch.log_softmax(F.linear(xs, ws), -1)
    log_p = torch.log_softmax(F.linear(xt, wt), -1)
    expect = (log_p.exp() * (log_p - log_q)).sum() / xs.shape[0]
    torch.testing.assert_close(reference(xs, ws, xt, wt, jsd_beta=0.0), expect, rtol=1e-13, atol=0)


# -------------------------------------------------------------------------------------- math facts
def test_symmetric_at_half():
    xs, ws, xt, wt = _inputs(Hs=6, Ht=6, seed=4)
    a = reference(xs, ws, xt, wt, jsd_beta=0.5)
    b = reference(xt, wt, xs, ws, jsd_beta=0.5)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=0)


@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_beta_swaps_roles(beta):
    # JSD_beta(P || Q) == JSD_{1-beta}(Q || P)
    xs, ws, xt, wt = _inputs(Hs=6, Ht=6, scale=3.0, seed=5)
    a = reference(xs, ws, xt, wt, jsd_beta=beta)
    b = reference(xt, wt, xs, ws, jsd_beta=1 - beta)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("beta", BETAS)
def test_zero_when_distributions_match(beta):
    xs, ws, _, _ = _inputs(seed=6)
    xs.requires_grad_(True)
    ws.requires_grad_(True)
    loss = reference(xs, ws, xs.detach().clone(), ws.detach().clone(), jsd_beta=beta)
    gx, gw = _grads(loss, xs, ws)
    assert abs(loss.item()) < 1e-14
    assert gx.abs().max() < 1e-14 and gw.abs().max() < 1e-14


@pytest.mark.parametrize("beta", [0.05, 0.3, 0.5, 0.8])
def test_bounds_nonnegative_and_binary_entropy(beta):
    # 0 <= JSD_beta <= h(beta) = -beta log beta - (1-beta) log(1-beta), the bound approached by disjoint supports
    xs, ws, xt, wt = _inputs(BT=8, V=50, scale=40.0, seed=8)
    per_row_mean = reference(xs, ws, xt, wt, jsd_beta=beta)
    h = -beta * math.log(beta) - (1 - beta) * math.log(1 - beta)
    assert 0.0 <= per_row_mean.item() <= h + 1e-12


def test_endpoint_switch_is_discontinuous():
    xs, ws, xt, wt = _inputs(scale=3.0, seed=9)
    near0 = reference(xs, ws, xt, wt, jsd_beta=1e-9).item()
    at0 = reference(xs, ws, xt, wt, jsd_beta=0.0).item()
    near1 = reference(xs, ws, xt, wt, jsd_beta=1 - 1e-9).item()
    at1 = reference(xs, ws, xt, wt, jsd_beta=1.0).item()
    assert near0 < 1e-7 and near1 < 1e-7
    assert at0 > 1e-3 and at1 > 1e-3


def test_large_logits_finite():
    # log-space mixture: no underflow/NaN for very peaked distributions
    xs, ws, xt, wt = _inputs(BT=4, V=300, scale=200.0, seed=10)
    xs.requires_grad_(True)
    ws.requires_grad_(True)
    for beta in BETAS:
        loss = reference(xs, ws, xt, wt, jsd_beta=beta)
        gx, gw = _grads(loss, xs, ws)
        assert torch.isfinite(loss) and torch.isfinite(gx).all() and torch.isfinite(gw).all()


def test_temperature_equals_scaling_inputs():
    xs, ws, xt, wt = _inputs(seed=11)
    for beta in BETAS:
        a = reference(xs, ws, xt, wt, jsd_beta=beta, temperature=2.5)
        b = reference(xs / 2.5, ws, xt / 2.5, wt, jsd_beta=beta, temperature=1.0)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-15)


# ------------------------------------------------------------------------------------- ignore_index
@pytest.mark.parametrize("beta", BETAS)
def test_ignore_equals_subset(beta):
    xs, ws, xt, wt = _inputs(BT=9, seed=12)
    lab = torch.tensor([3, -100, 0, 10, -100, -100, 7, 1, 2])
    keep = lab != -100
    a = reference(xs, ws, xt, wt, shift_labels=lab, jsd_beta=beta)
    b = reference(xs[keep], ws, xt[keep], wt, jsd_beta=beta)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-15)


def test_ignore_index_may_be_a_vocab_id_and_label_values_do_not_matter():
    xs, ws, xt, wt = _inputs(BT=6, V=11, seed=13)
    lab1 = torch.tensor([4, 2, 2, 9, 0, 2])
    lab2 = torch.tensor([0, 2, 2, 10, 5, 2])  # same rows equal to ignore_index=2, other values changed
    a = reference(xs, ws, xt, wt, shift_labels=lab1, ignore_index=2)
    b = reference(xs, ws, xt, wt, shift_labels=lab2, ignore_index=2)
    c = reference(xs[[0, 3, 4]], ws, xt[[0, 3, 4]], wt)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, c, rtol=1e-12, atol=1e-15)


def test_ignored_rows_get_zero_gradient_and_all_ignored_is_zero():
    xs, ws, xt, wt = _inputs(BT=5, seed=14)
    xs.requires_grad_(True)
    ws.requires_grad_(True)
    lab = torch.tensor([1, -100, 2, -100, 3])
    gx, _ = _grads(reference(xs, ws, xt, wt, shift_labels=lab), xs, ws)
    assert torch.all(gx[1] == 0) and torch.all(gx[3] == 0) and gx[0].abs().max() > 0
    loss = reference(xs, ws, xt, wt, shift_labels=torch.full((5,), -100))
    gx, gw = _grads(loss, xs, ws)
    assert loss.item() == 0.0 and torch.all(gx == 0) and torch.all(gw == 0)


# ---------------------------------------------------------------------------------- contract details
def test_teacher_is_constant_and_accum_dtype_is_ignored():
    xs, ws, xt, wt = _inputs(seed=15)
    for t in (xs, ws, xt, wt):
        t.requires_grad_(True)
    a = reference(xs, ws, xt, wt, jsd_beta=0.3, accum_dtype=None)
    b = reference(xs, ws, xt, wt, jsd_beta=0.3, accum_dtype=torch.float32)
    assert torch.equal(a, b)
    a.backward()
    assert xs.grad is not None and ws.grad is not None
    assert xt.grad is None and wt.grad is None


def test_upstream_gradient_scales_linearly():
    xs, ws, xt, wt = _inputs(seed=16)
    xs.requires_grad_(True)
    ws.requires_grad_(True)
    gx1, gw1 = _grads(reference(xs, ws, xt, wt, jsd_beta=0.4), xs, ws)
    gx2, gw2 = _grads(reference(xs, ws, xt, wt, jsd_beta=0.4) * -3.25, xs, ws)
    torch.testing.assert_close(gx2, -3.25 * gx1, rtol=1e-13, atol=0)
    torch.testing.assert_close(gw2, -3.25 * gw1, rtol=1e-13, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_computes_in_input_dtype(dtype):
    xs, ws, xt, wt = _inputs(seed=17, dtype=dtype)
    xs.requires_grad_(True)
    loss = reference(xs, ws, xt, wt, shift_labels=torch.tensor([0, -100, 1, 2, 3]))
    assert loss.dtype == dtype and loss.dim() == 0
    (gx,) = torch.autograd.grad(loss, xs)
    assert gx.dtype == dtype


def test_float32_close_to_float64():
    xs, ws, xt, wt = _inputs(BT=16, Hs=32, Ht=24, V=97, scale=3.0, seed=18)
    for beta in BETAS:
        a = reference(xs, ws, xt, wt, jsd_beta=beta, temperature=1.5)
        b = reference(*(t.float() for t in (xs, ws, xt, wt)), jsd_beta=beta, temperature=1.5)
        assert abs(a.item() - b.item()) <= 1e-5 * abs(a.item()) + 1e-7
