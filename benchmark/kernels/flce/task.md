# Fused linear cross entropy (Liger-Kernel `liger_fused_linear_cross_entropy`)

## What the op computes
Inputs: hidden states `input` X (BT x H), an output projection `weight` W (V x H), an optional `bias` b (V),
integer class `target` y (BT). Logits Z = X W^T + b (BT x V). The op returns the cross-entropy loss of Z
without materializing Z in full (it processes the tokens in chunks).
For every token i with y_i != ignore_index, with lse_i = log sum_c exp(Z_ic), eps = label_smoothing and
s = lse_square_scale:

    l_i = (1 - eps) * (lse_i - Z_{i,y_i})  +  (eps / V) * sum_c (lse_i - Z_ic)  +  s * lse_i^2

Tokens with y_i == ignore_index have l_i = 0 and contribute nothing to any gradient.
- reduction="none": the vector (l_1, ..., l_BT), shape (BT,).
- reduction="sum": sum_i l_i, shape ().
- reduction="mean": sum_i l_i / N, N = number of non-ignored tokens (the z-loss term is divided by N as well);
  if every token is ignored the loss is 0 (and all gradients are 0) for every reduction.

## Entry point and implementation
`task/entry.py: kernel(input, weight, target, bias=None, ignore_index=-100, lse_square_scale=0.0,
label_smoothing=0.0, reduction="mean")` returns the loss tensor. It calls
`liger_kernel.transformers.functional.liger_fused_linear_cross_entropy` with keyword arguments. Source:
- `liger/src/liger_kernel/transformers/functional.py` (the public wrapper)
- `liger/src/liger_kernel/ops/fused_linear_cross_entropy.py` (chunked forward, gradient handling, the
  `torch.autograd.Function` `LigerFusedLinearCrossEntropyFunction`)
- `liger/src/liger_kernel/ops/cross_entropy.py` (Triton kernel `liger_cross_entropy_kernel`: per-token loss and
  logit gradients) and `liger/src/liger_kernel/ops/utils.py` (helper Triton kernels)

## Arguments
| argument | shape | dtype | domain | differentiable |
|---|---|---|---|---|
| input | (BT, H), BT >= 1, H >= 1 | float32 | finite | yes |
| weight | (V, H), V >= 1 | float32 | finite | yes, or frozen |
| target | (BT,) | int64 | each entry in [0, V) or == ignore_index; any number ignored (none to all) | no |
| bias | (V,) or None | float32 | finite | yes, or frozen |
| ignore_index | int | | any int, e.g. -100 or a class id in [0, V) | |
| lse_square_scale | float | | >= 0 | |
| label_smoothing | float | | [0, 1] | |
| reduction | str | | "mean", "sum", "none" | |

Sizes need not be powers of two. Every tensor argument may have any strides and storage offset (it need not be
contiguous).

## Output
One tensor: shape () for "mean"/"sum", (BT,) for "none", float32. Inputs are never modified.

## Gradients
Backward accepts any finite upstream gradient g (`loss.backward(g)`): a scalar for "mean"/"sum", a (BT,)
tensor for "none". Gradients flow to input, weight and bias through Z: with G = d<g, loss>/dZ (BT x V),
dX = G W, dW = G^T X, db = sum_i G_i, where for a non-ignored token
dl_i/dZ_ic = (1 + 2 s lse_i) softmax(Z_i)_c - (1 - eps) [c == y_i] - eps / V (and 0 for an ignored token).
target receives no gradient.
`input` must require grad whenever a gradient is taken; `weight` and `bias` may each require grad or be frozen
(requires_grad=False). Forward-only calls (no tensor requires grad) are in scope for the loss value.

## Scope
In scope: everything in the tables above: bias present or absent; reduction mean/sum/none; ignore_index with
none, some or all targets ignored; label_smoothing; lse_square_scale (z-loss); trainable or frozen weight/bias.
Out of scope (the entry point does not expose them; they stay at their defaults): `ce_weight` (per-class
weights), `softcap`, `return_z_loss`, `return_token_accuracy`, `return_predicted_tokens`, `accum_dtype`,
`use_token_scaling` and any other keyword of the wrapper; bfloat16, float16, float64 or mixed-dtype inputs;
autocast/AMP; calls
where `input` does not require grad but `weight` or `bias` does; calling backward twice on one graph
(`retain_graph=True`); double backward, `torch.func` transforms and `torch.compile`.

## Oracle and running
`task/reference.py: reference(...)` has the same signature and output structure and is plain PyTorch. Its
float64 autograd (all floating inputs cast to float64, target unchanged) is the trusted oracle for the loss
and for the gradients of input, weight and bias. In float32 the op normally agrees with the oracle to about 1e-6
of the quantity's natural scale: max(max|oracle|, per-token scale), where the per-token scale is |g|/N for the
loss-gradient entries (N = non-ignored tokens under "mean", else 1) times max|W| for dX or max|X| for dW. Use that
floor: a loss or gradient that is tiny only because O(1) terms cancel (e.g. one very confident correct token) is
not meaningfully compared relative to its own magnitude.
Without a GPU, run on CPU tensors with `TRITON_INTERPRET=1` (Triton interpreter; slow, so sizes of a few hundred
are practical); on a GPU use CUDA tensors. `liger/` is read-only; `scratch/` is writable.
