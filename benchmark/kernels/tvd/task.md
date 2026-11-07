# TVD loss (Liger-Kernel `LigerTVDLossFunction`)

Total Variation Distance loss: compares a student distribution `p` with a target (teacher) `q`, row by row.

## What the op computes

`p` and `q` have shape `(BT, V)` (BT rows, e.g. batch x sequence positions; V columns, e.g. vocabulary).
Per element: `d[i, j] = 0.5 * |p[i, j] - q[i, j]|`.
Row mask: `m[i] = 0` if `shift_labels` is given and `shift_labels[i] == ignore_index`, else `m[i] = 1`.
`N = sum_i m[i]` is the number of kept rows (`N = BT` when `shift_labels` is None).

| `reduction`   | loss                            | loss shape |
|---------------|---------------------------------|------------|
| `"none"`      | `L[i, j] = m[i] * d[i, j]`      | `(BT, V)`  |
| `"sum"`       | `sum_ij m[i] * d[i, j]`         | `()`       |
| `"mean"`      | `sum_ij m[i] * d[i, j] / (N*V)` | `()`       |
| `"batchmean"` | `sum_ij m[i] * d[i, j] / N`     | `()`       |

When each row of `p` and `q` is a probability distribution, `sum_j d[i, j]` is the total variation distance
between them, a value in `[0, 1]`. The op itself is elementwise and accepts any finite values: rows need not be
normalized or non-negative.

## Entry point and implementation

- `task/entry.py`: `kernel(p, q, shift_labels=None, reduction="batchmean", ignore_index=-100)` calls
  `liger_kernel.transformers.functional.liger_tvd` with keyword arguments (its own default reduction is `"mean"`,
  so the entry always passes `reduction`). That calls `LigerTVDLossFunction.apply(p, q, shift_labels, reduction, ignore_index)`.
- `liger/src/liger_kernel/ops/tvd.py`: the Triton kernel `_tv_distance_kernel`, its launcher `tv_distance_forward_triton`,
  `tvd_backward_triton`, and the autograd Function `LigerTVDLossFunction`, whose forward and backward are wrapped by
  `ensure_contiguous` (`liger/src/liger_kernel/ops/utils.py`).
- `liger/src/liger_kernel/transformers/functional.py` (`liger_tvd`) and `liger/src/liger_kernel/transformers/tvd.py`
  (module `LigerTVDLoss`, default reduction `"batchmean"`) are thin wrappers around the Function.
- `liger_kernel.ops` re-exports the Function. On CPU and CUDA it is the one in `ops/tvd.py`.

## Arguments

| argument       | shape                      | dtype                                       | domain                                          | differentiable |
|----------------|----------------------------|---------------------------------------------|-------------------------------------------------|----------------|
| `p`            | `(BT, V)`, BT >= 1, V >= 1 | float32 or bfloat16                         | any finite values                               | yes            |
| `q`            | same shape as `p`          | float32 or bfloat16 (may differ from `p`'s) | any finite values                               | no: constant target, `q.grad` stays None even if `q.requires_grad` |
| `shift_labels` | `(BT,)` or None            | int64                                       | any integers; only compared with `ignore_index` | no             |
| `reduction`    | str                        |                                             | `"none"`, `"sum"`, `"mean"`, `"batchmean"`      |                |
| `ignore_index` | int                        |                                             | any integer (default -100)                      |                |

Tensors are passed to the op as given (contiguous tensors, views and arbitrary strides are all valid); the op does not modify `p` or `q`.

## Outputs and gradients

One tensor, `loss`, with the shape given in the table above. The op returns `loss` as float32 for every input
dtype. `reference()` returns its inputs' (promoted) floating dtype, so compare values, not dtypes.
Gradients flow to `p` only, and `p.grad` has `p`'s dtype. With `reduction="none"`, backward takes an upstream gradient
of shape `(BT, V)`. Otherwise it takes a scalar upstream gradient (1.0 from `loss.backward()`, or any other finite
value). Rows with `m[i] = 0` contribute to neither the loss nor `N`. The gradient with respect to `p` is defined
as the float64 autograd gradient of `task/reference.py` (see Oracle).

## Scope

In scope:
- all four reductions, each with `shift_labels` None or given (any subset of rows ignored; for `"mean"` and
  `"batchmean"` at least one row must be kept), and any `ignore_index`;
- dtype pairs `(p, q)` in {(float32, float32), (bfloat16, float32), (float32, bfloat16)};
- contiguous or strided inputs; any BT >= 1 and V >= 1 (V need not be a power of 2 and may exceed the kernel's block size);
- scalar upstream gradients for reduced losses, and `(BT, V)` upstream gradients for `"none"`.

Out of scope:
- `p` and `q` both bfloat16. The Triton CPU interpreter (`TRITON_INTERPRET=1`, which runs these kernels here)
  computes bfloat16-with-bfloat16 arithmetic incorrectly, at every commit.
- float16 or float64 inputs to the op, non-finite values (inf/NaN), BT = 0 or V = 0, and non-int64 `shift_labels`.
- Every row ignored with `"mean"` or `"batchmean"`. The op raises ZeroDivisionError and `reference()` raises ValueError.
- Double backward; other devices' implementations (`liger_kernel/ops/backends/_ascend`); the separate fused
  cross-entropy + TVD op (`ops/fused_ce_tvd.py`, `LigerFusedCETVDLoss`).

## Oracle

`task/reference.py` defines `reference()`, with the same signature and output structure as `kernel()`: a plain-PyTorch
implementation of the math above. Its autograd evaluated in float64 (every floating input cast to float64,
`shift_labels` unchanged) is the trusted oracle for the loss and for the gradient with respect to `p`.
Expected agreement of the kernel (native dtype) with the oracle: about 1e-6 relative or better for the loss; for
`p.grad`, about 1e-6 relative when `p` is float32 and up to about 2e-2 relative when `p` is bfloat16 (the gradient is
stored in bfloat16, the CPU interpreter truncates on that store, and a non-unit upstream adds rounding). `scratch/` is writable.
