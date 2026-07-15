"""Fused CUDA WKV kernel, used by RWKVTimeMix.forward (mini_vlm/models/rwkv.py)
as a drop-in replacement for its pure-Python `for t in range(T)` recurrence
when running on GPU. The kernel source under mini_vlm/models/cuda/ is ported
from BlinkDL/RWKV-LM (Apache-2.0) -- see the headers in those files.

The kernel is JIT-compiled on first use via torch.utils.cpp_extension.load,
which needs a CUDA-capable PyTorch build and a working nvcc toolchain. This
repo's CPU-only dev machine (see CLAUDE.md) has neither, so `available()`
always returns False here and RWKVTimeMix silently keeps using its existing
Python loop -- nothing about the CPU path changes. On a CUDA machine, run
mini_vlm/verify_wkv_cuda.py once to confirm this kernel and the Python loop
agree numerically before trusting it for real training.
"""
import os

import torch

_CUDA_DIR = os.path.join(os.path.dirname(__file__), "cuda")

# Must be >= the longest sequence length any RWKV stack in this project
# processes: the spatial bridge's H*W=49 (RWKVSpatialBridge, spatial_size=7)
# and the language core's n_compressed_tokens + max_question_len = 8 + 16 = 24
# (RWKVLanguageCore). 64 leaves headroom for both without wasting much of
# the backward kernel's per-thread local `F y[Tmax], z[Tmax], zexp[Tmax]`.
T_MAX = 64

_kernel = None
_load_failed = False


def _load_kernel():
    global _kernel, _load_failed
    if _kernel is not None or _load_failed:
        return _kernel
    try:
        from torch.utils.cpp_extension import load

        _kernel = load(
            name="mini_vlm_wkv",
            sources=[
                os.path.join(_CUDA_DIR, "wkv_op.cpp"),
                os.path.join(_CUDA_DIR, "wkv_cuda.cu"),
            ],
            extra_cuda_cflags=["--use_fast_math", "-O3", f"-DTmax={T_MAX}"],
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
        print(f"[mini_vlm] WKV CUDA kernel failed to compile, falling back to the "
              f"Python-loop WKV implementation: {exc}")
        _load_failed = True
        _kernel = None
    return _kernel


def available(x: torch.Tensor) -> bool:
    """Whether the CUDA WKV path should be used for a given input tensor.

    False on CPU (including this repo's dev machine) or if compilation
    failed; RWKVTimeMix falls back to its Python loop in either case.
    """
    if not x.is_cuda:
        return False
    if os.environ.get("MINI_VLM_WKV_CUDA", "1") == "0":
        return False
    return _load_kernel() is not None


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, time_decay, time_first, k, v):
        B, T, C = k.shape
        assert T <= T_MAX, f"sequence length {T} exceeds T_MAX={T_MAX}"
        in_dtype = k.dtype
        w = (-torch.exp(time_decay.float())).contiguous()
        u = time_first.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device=k.device, dtype=torch.float32,
                         memory_format=torch.contiguous_format)
        _kernel.forward(B, T, C, w, u, k, v, y)
        return y.to(in_dtype)

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        B, T, C = k.shape
        gw = torch.zeros((B, C), device=k.device, dtype=torch.float32).contiguous()
        gu = torch.zeros((B, C), device=k.device, dtype=torch.float32).contiguous()
        gk = torch.zeros((B, T, C), device=k.device, dtype=torch.float32).contiguous()
        gv = torch.zeros((B, T, C), device=k.device, dtype=torch.float32).contiguous()
        _kernel.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return gw, gu, gk, gv


def run_cuda_wkv(time_decay: torch.Tensor, time_first: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """CUDA-kernel WKV recurrence. `time_decay`/`time_first` are the raw
    (pre -exp) per-channel parameters, matching RWKVTimeMix's own
    self.time_decay / self.time_first -- the -exp(time_decay) transform
    happens inside WKV.forward, same convention as upstream RWKV-LM."""
    return WKV.apply(time_decay, time_first, k, v)
