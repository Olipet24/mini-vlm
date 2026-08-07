"""Small trainable transform applied to the frozen vision encoder's cached
[B, 576, 7, 7] feature tensor, identically for both the primary and baseline
models. Owned by train.py (not either model class) and applied at the same
point in both models' forward paths, so it can never drift into a primary-
specific gadget -- this is the shared/fairness-mirrored lever standing in
for "fine-tune the vision backbone" (Pythia's biggest single lever over the
bottom-up-top-down baseline, per report/final_report_v2.tex's Background
section) without abandoning the offline-feature-cache design: it only ever
sees the already-cached tensor, never a raw image, so training stays exactly
as cheap as before.

Both modes are initialized to the identity transform, so enabling this with
fresh random init doesn't perturb the epoch-0 loss relative to --feature-
adapter none -- any effect comes from what training does with it, not from
a different starting point.
"""
import torch
import torch.nn as nn


class CachedFeatureAdapter(nn.Module):
    def __init__(self, channels: int = 576, mode: str = "affine") -> None:
        super().__init__()
        assert mode in ("affine", "conv1x1")
        self.mode = mode
        if mode == "affine":
            self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.conv = nn.Conv2d(channels, channels, kernel_size=1)
            with torch.no_grad():
                self.conv.weight.copy_(torch.eye(channels).view(channels, channels, 1, 1))
                self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "affine":
            return x * self.scale + self.shift
        return self.conv(x)
