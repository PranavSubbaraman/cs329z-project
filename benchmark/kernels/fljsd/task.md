# fljsd: fused linear + generalized Jensen-Shannon divergence (distillation loss)

## What the op computes
Student hidden states `X_s` (BT x H_s) with student head `W_s` (V x H_s); teacher hidden states `X_t` (BT x H_t)
with teacher head `W_t` (V x H_t); temperature `T`; mixing coefficient `beta`. For every row `i`:

    z_s = X_s W_s^T / T,  z_t = X_t W_t^T / T                       (BT x V logits)
    log q_i = log_softmax(z_s[i]),  log p_i = log_softmax(z_t[i])   (Q_i = student, P_i = teacher)
    beta == 0      : d_i = KL(P_i || Q_i) = sum_v p (log p - log q)        (forward KL)
    beta == 1      : d_i = KL(Q_i || P_i) = sum_v q (log q - log p)        (reverse KL)
    0 < beta < 1   : M_i = beta P_i + (1 - beta) Q_i
                     d_i = beta KL(P_i || M_i) + (1 - beta) KL(Q_i || M_i)
    loss = (1 / N) * sum_{i kept} d_i

Row `i` is kept unless `shift_labels[i] == ignore_index` (every row is kept when `shift_labels` is None).
`N` is the number of kept rows; if `N == 0` the loss is 0 and all gradients are 0. The endpoints are exact
switches: at exactly 0 and exactly 1 the op uses the KL formulas above (the interior formula tends to 0 there).
No `T^2` factor is applied.

The op never materializes the full BT x V logits: it walks over chunks of rows, computes each chunk's loss
contribution and the gradients with respect to the student tensors during the forward pass, and stores the
gradients for the backward pass.

## Entry point and implementation
- `task/entry.py: kernel(...)` calls `liger_kernel.transformers.functional.liger_fused_linear_jsd`, which applies
  `LigerFusedLinearJSDFunction` (`liger/src/liger_kernel/ops/fused_linear_jsd.py`: chunked forward,
  `fused_linear_jsd_forward`, and `fused_linear_jsd_backward`).
- Per-chunk divergence + gradient: op `jsd_loss_and_grad` through the backend dispatcher
  (`liger/src/liger_kernel/backends/`, op locations declared in `liger/src/liger_kernel/functional.py`). Triton impl:
  `liger/src/liger_kernel/ops/backends/_triton/jsd.py`, kernel `_jsd_kernel` in `liger/src/liger_kernel/ops/jsd.py`.
  CuTe DSL / cuTile impls (`ops/backends/_cutedsl/`, `ops/backends/_cutile/`) need CUDA packages; without them the
  dispatcher uses Triton.
- Backward scaling by the upstream gradient: `element_mul_kernel` in `liger/src/liger_kernel/ops/utils.py`.
- Module form of the same op: `LigerFusedLinearJSD` in `liger/src/liger_kernel/transformers/fused_linear_jsd.py`.

## Arguments (all passed by keyword)
| argument | shape | dtype | domain | differentiable |
|---|---|---|---|---|
| `student_input` | (BT, H_s) | float32, bfloat16 or float16 | finite | yes |
| `student_weight` | (V, H_s) | same as student_input | finite | yes |
| `teacher_input` | (BT, H_t) | same | finite | no (constant) |
| `teacher_weight` | (V, H_t) | same | finite | no (constant) |
| `shift_labels` | (BT,) or None | int64 | each entry in [0, V-1] or equal to `ignore_index` | no |
| `jsd_beta` | python float | | [0, 1], endpoints included; default 0.5 | no |
| `ignore_index` | python int | | any int (may coincide with a vocab id); default -100 | no |
| `temperature` | python float | | > 0; default 1.0 | no |
| `accum_dtype` | None or torch.float32 | | dtype of the internal grad_weight accumulation buffer; default None | no |

BT >= 1, V >= 2, H_s >= 1, H_t >= 1; any sizes (powers of two not required); H_s and H_t may differ. All four
floating tensors share one dtype and one device. Every tensor has unit stride along its last dimension (row slices
and other row-strided views are allowed).

## Outputs
`loss`: a 0-dim tensor (the op returns it in float32 for every input dtype). `loss.backward(g)` for any finite scalar
`g` fills `student_input.grad` and `student_weight.grad` (same shape and dtype as those inputs, scaled by `g`).
`teacher_input.grad` and `teacher_weight.grad` stay None even if they require grad; ignored rows get zero gradient.
The op does not modify any of its inputs.

## Gradient flow
Let `G = dL/d(X_s W_s^T)` (BT x V, before the 1/T scaling; zero rows for ignored tokens). Then `dL/dX_s = G W_s`,
`dL/dW_s = G^T X_s`, and `G[i] = (g_i - q_i * sum_v g_i) / T`, where `g_i` is the partial derivative of `d_i / N`
with respect to the vector `log q_i` (teacher fixed). Nothing flows to the teacher tensors.

## Scope
In scope: every combination above, including beta = 0, beta = 1 and interior values; temperature != 1; labels
None, partly ignored or all ignored; accum_dtype None or float32; H_s != H_t; any upstream gradient `g`.
Out of scope: calling inside `torch.autocast`; float64 inputs or mixed dtypes; tensors whose last dimension is not
unit-stride (for example transposed views); gradients with respect to the teacher tensors or `shift_labels`; double
backward; explicit backend selection (`jsd_impl`/`jsd_mode`, not exposed by the entry point).

## Oracle
`task/reference.py: reference(...)` has the same signature and output. It is the plain-PyTorch reference: run with
every floating input cast to float64, its loss and its autograd gradients are the trusted oracle. Results for
bfloat16/float16 inputs are judged against it with the oracle's dtype-aware tolerance rule (the same rule for
every task; it is documented with the oracle helpers).
