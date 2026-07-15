"""The RWKV paper identifies that faster convergence can be obtained by increasing initializing the embedding to be in the following
"""
import math

import torch
import torch.nn as nn


def rwkv_init(model: nn.Module) -> None:
    """In-place orthogonal (or zero) init of every Linear/Embedding weight
    reachable from `model`, honoring per-module `scale_init` overrides."""
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            ww = m.weight
            gain = math.sqrt(max(ww.shape[0], ww.shape[1]))
            scale = getattr(m, "scale_init", 1e-4)
        elif isinstance(m, nn.Linear):
            ww = m.weight
            gain = math.sqrt(ww.shape[0] / ww.shape[1]) if ww.shape[0] > ww.shape[1] else 1.0
            scale = getattr(m, "scale_init", 1.0)
        else:
            continue

        with torch.no_grad():
            gain *= scale
            if gain == 0:
                nn.init.zeros_(ww)
            else:
                nn.init.orthogonal_(ww, gain=gain)
