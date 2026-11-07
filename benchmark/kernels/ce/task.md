# Kernel `ce`: cross-entropy loss (Liger-Kernel)

Liger-Kernel's Triton cross-entropy loss over a batch of logit rows, with optional label smoothing and an
optional z-loss (squared log-sum-exp) term. Tensors are on CPU (Triton interpreter, `TRITON_INTERPRET=1`) or CUDA.

## Entry point and implementation
- `task/entry.py: kernel(input, target, ignore_index=-100, reduction="mean", label_smoothing=0.0,
  lse_square_scale=0.0)` passes its arguments unchanged, as keywords, to
  `liger_kernel.transformers.functional.liger_cross_entropy` (class weights and softcap off, extra outputs off),
  which calls the autograd function `LigerCrossEntropyFunction`.
- Source: `liger/src/liger_kernel/ops/cross_entropy.py` (Triton kernel `liger_cross_entropy_kernel`, host code
  `cross_entropy_forward` / `cross_entropy_backward`, `LigerCrossEntropyFunction`),
  `liger/src/liger_kernel/ops/utils.py` (`element_mul_kernel`, used by the backward pass) and
  `liger/src/liger_kernel/transformers/functional.py` (`liger_cross_entropy`).

## What it computes
Row i has logits x_i (length V) and target y_i. Let lse_i = log sum_c exp(x_ic), log p_ic = x_ic - lse_i,
e = label_smoothing, s = lse_square_scale. Row i is *counted* iff y_i != ignore_index.

    l_i = (1 - e) * (-log p_{i,y_i})  +  e * (1/V) * sum_c (-log p_ic)  +  s * lse_i^2    (counted rows)
    l_i = 0                                                                              (other rows)

- `reduction="none"`: the vector (l_1, ..., l_BT).
- `reduction="sum"`: sum_i l_i.
- `reduction="mean"`: sum_i l_i / N, where N = number of counted rows (the result is 0 when N = 0).

With s = 0 this is `torch.nn.functional.cross_entropy(input, target, ignore_index=ignore_index,
reduction=reduction, label_smoothing=e)`. The z-term s * lse_i^2 is added on counted rows only and, under
"mean", is divided by the same N.

## Arguments
| name | shape / type | dtype | domain | differentiable |
|---|---|---|---|---|
| `input` | tensor (BT, V), BT >= 1, V >= 2 | float32 or float16 | finite logits; last dimension unit-stride (`stride(-1) == 1`); rows may be strided (e.g. a column slice of a wider tensor); elements must not overlap in memory | yes (when it requires grad) |
| `target` | tensor (BT,) | int64 | every value in [0, V-1] or equal to `ignore_index`; any stride | no (constant) |
| `ignore_index` | int | - | any integer (default -100); may lie inside [0, V-1] | - |
| `reduction` | str | - | `"mean"` (default), `"sum"`, `"none"` | - |
| `label_smoothing` | float | - | 0 <= e <= 1 (default 0.0) | - |
| `lse_square_scale` | float | - | s >= 0 (default 0.0) | - |

## Output
A single tensor with `input`'s dtype and device: shape `()` for "mean" and "sum", shape `(BT,)` for "none".
In scope only where the exact result is representable in that dtype (a float16 "sum" over many rows can exceed
the float16 range).

## Scope
In scope: every combination of argument values allowed by the table above.
Out of scope (not exercised by this benchmark): class weights (`weight`), `softcap`, the extra outputs
(`return_z_loss`, `return_token_accuracy`, `return_predicted_tokens`), the legacy `size_average` / `reduce`
placeholders, bfloat16 or float64 `input`, `input` with `stride(-1) != 1`, target values outside [0, V-1] other
than `ignore_index` (Liger raises an AssertionError), non-finite logits, the `nn.Module` wrapper
`LigerCrossEntropyLoss` and alternative implementations selected through `LIGER_KERNEL_IMPL`.

## How gradients flow
Only `input` receives a gradient; `target` is a constant. For counted rows

    d l_i / d x_ij = p_ij - (1 - e) * [j == y_i] - e / V + 2 * s * lse_i * p_ij ,

ignored rows get 0, the whole gradient is divided by N under "mean", and it is multiplied by the incoming
gradient (a scalar for "mean"/"sum", one value per row for "none").
Side effects (documented design): when `input.requires_grad` is True, Liger computes the gradient during the
forward pass and stores it in `input`'s own storage to save memory, so after the call `input` holds the gradient
rather than the logits, and for "mean"/"sum" `input.grad` may share that storage. That overwrite is intended.
`target` is never modified.

## Reference (oracle)
`task/reference.py: reference(...)` has the same signature and output structure as `kernel(...)` and is plain
PyTorch, differentiable through autograd. Evaluated with float64 inputs, its value and its autograd gradient
with respect to `input` are the trusted ground truth.
