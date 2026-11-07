"""HARNESS-ONLY feasibility probe: import + tiny fwd/bwd of each Liger op under TRITON_INTERPRET=1 on CPU.
Usage: TRITON_INTERPRET=1 PYTHONPATH=work/trees/<commit>/src python scripts/harness/feasibility.py <kernel>"""
import sys, time, torch
k = sys.argv[1]
torch.manual_seed(0)
t0 = time.time()
if k == "tvd":
    from liger_kernel.ops.tvd import LigerTVDLossFunction as F
    p = torch.randn(4, 64).softmax(-1).requires_grad_(); q = torch.randn(4, 64).softmax(-1)
    F.apply(p, q, None, "batchmean", -100).backward(); g = p.grad
elif k == "flce":
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction as F
    x = torch.randn(6, 16, requires_grad=True); w = torch.randn(32, 16, requires_grad=True); y = torch.randint(0, 32, (6,))
    out = F.apply(x, w, y); loss = out[0] if isinstance(out, tuple) else out
    loss.backward(); g = w.grad
elif k == "fljsd":
    from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction as F
    xs = torch.randn(6, 16, requires_grad=True); ws = torch.randn(32, 16, requires_grad=True)
    xt = torch.randn(6, 16); wt = torch.randn(32, 16)
    F.apply(xs, ws, xt, wt).backward(); g = ws.grad
elif k == "ce":
    from liger_kernel.transformers.functional import liger_cross_entropy
    x = torch.randn(6, 32, requires_grad=True); y = torch.randint(0, 32, (6,))
    out = liger_cross_entropy(x, y); loss = out[0] if isinstance(out, tuple) else out
    loss.backward(); g = x.grad
print(f"{k}: OK grad_norm={g.norm().item():.4f} {time.time()-t0:.2f}s")
