"""Check that the CUDA WKV kernel (mini_vlm/models/wkv_cuda_kernel.py,
ported from BlinkDL/RWKV-LM) agrees numerically with RWKVTimeMix's existing
pure-Python loop, on both output and gradients, before trusting it for real
training.

Must be run on a CUDA machine -- this repo's CPU-only dev machine can't
compile or exercise the kernel at all (wkv_cuda_kernel.available() is always
False there, so RWKVTimeMix silently stays on the Python loop and this
script has nothing to compare).

Usage:
    python -m mini_vlm.verify_wkv_cuda
"""
import os

import torch

from mini_vlm.models.rwkv import RWKVTimeMix


def run_once(model: RWKVTimeMix, x: torch.Tensor, use_cuda_kernel: bool):
    os.environ["MINI_VLM_WKV_CUDA"] = "1" if use_cuda_kernel else "0"
    model.zero_grad()
    xi = x.clone().detach().requires_grad_(True)
    out = model(xi)
    out.sum().backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return out.detach(), xi.grad.clone(), grads


def check(label: str, B: int, T: int, C: int, device: torch.device):
    torch.manual_seed(0)
    model = RWKVTimeMix(n_embd=C, layer_id=0, n_layer=4).to(device)
    x = torch.randn(B, T, C, device=device)

    out_cuda, xgrad_cuda, grads_cuda = run_once(model, x, use_cuda_kernel=True)
    out_loop, xgrad_loop, grads_loop = run_once(model, x, use_cuda_kernel=False)

    out_diff = (out_cuda - out_loop).abs().max().item()
    xgrad_diff = (xgrad_cuda - xgrad_loop).abs().max().item()
    print(f"[{label}] B={B} T={T} C={C}  max|out diff|={out_diff:.2e}  max|dL/dx diff|={xgrad_diff:.2e}")

    ok = out_diff < 1e-3 and xgrad_diff < 1e-3
    for name in grads_cuda:
        diff = (grads_cuda[name] - grads_loop[name]).abs().max().item()
        ok = ok and diff < 1e-3
        print(f"    max|d{name} diff|={diff:.2e}")
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- run this on the GPU machine.")

    device = torch.device("cuda")
    from mini_vlm.models import wkv_cuda_kernel
    if wkv_cuda_kernel._load_kernel() is None:
        raise SystemExit("WKV CUDA kernel failed to compile -- see the error above.")

    results = [
        check("bridge-shaped (7x7=49 tokens)", B=4, T=49, C=128, device=device),
        check("core-shaped (8+16=24 tokens)", B=4, T=24, C=128, device=device),
    ]
    if all(results):
        print("\nOK: CUDA kernel matches the Python-loop WKV implementation.")
    else:
        raise SystemExit("\nMISMATCH: CUDA kernel disagrees with the Python-loop WKV implementation.")


if __name__ == "__main__":
    main()
