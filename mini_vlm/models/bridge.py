"""RWKV Spatial Bridge: the project's main architectural contribution.

Takes the frozen vision encoder's [C, 7, 7] spatial feature map,
flattens it into a 49-token sequence, runs it through a small RWKV
stack (linear-time, no attention), and compresses the 49 tokens down to
a small fixed number of language-aligned tokens via a *learned linear
pooling* matrix. This intentionally avoids cross-attention: pooling
weights are directly learned parameters (O(K x 49)), not dynamically
computed query/key dot products, which is what "no heavy cross-attention
matrices" means in the proposal.
"""
import torch
import torch.nn as nn

from mini_vlm.models.rwkv import RWKVStack


class RWKVSpatialBridge(nn.Module):
    def __init__(
        self,
        in_channels: int = 576,
        spatial_size: int = 7,
        d_model: int = 128,
        n_layer: int = 2,
        n_compressed_tokens: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        n_tokens = spatial_size * spatial_size

        self.in_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.rwkv = RWKVStack(n_embd=d_model, n_layer=n_layer, dropout=dropout)

        self.n_compressed_tokens = n_compressed_tokens
        self.pool_logits = nn.Parameter(torch.randn(n_compressed_tokens, n_tokens) * 0.02)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """feature_map: [B, C, H, W] -> visual tokens: [B, K, d_model]"""
        B, C, H, W = feature_map.shape
        x = feature_map.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.in_proj(x) + self.pos_embed
        x = self.rwkv(x)  # [B, H*W, d_model]

        pool = torch.softmax(self.pool_logits, dim=-1)  # [K, H*W]
        compressed = torch.einsum("kt,btd->bkd", pool, x)  # [B, K, d_model]
        return self.out_norm(compressed)
