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
        n_pool_heads: int = 1,
        add_global_token: bool = False,
        question_dim: int = 0,
    ) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        n_tokens = spatial_size * spatial_size

        self.in_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.rwkv = RWKVStack(n_embd=d_model, n_layer=n_layer, dropout=dropout)

        self.n_compressed_tokens = n_compressed_tokens
        self.n_pool_heads = n_pool_heads
        if n_pool_heads == 1:
            # Exact original param name/shape -- checkpoints trained before this
            # flag existed still load with strict=True.
            self.pool_logits = nn.Parameter(torch.randn(n_compressed_tokens, n_tokens) * 0.02)
        else:
            # H independent learned pooling matrices, concatenated and merged
            # back to d_model -- more expressive compression at the same K,
            # shaped like multi-head attention's concat+output-projection but
            # weights stay directly-learned nn.Parameters (no query/key dot
            # products), so this is still "not cross-attention".
            self.multihead_pool_logits = nn.Parameter(
                torch.randn(n_pool_heads, n_compressed_tokens, n_tokens) * 0.02
            )
            self.pool_merge = nn.Linear(n_pool_heads * d_model, d_model)

        # Question-conditioned pooling -- a linear-time, non-attention approximation of
        # co-attention's "which regions matter depends on the question" benefit: the K
        # pooling rows stay directly-learned parameters (pool_logits, the base pattern),
        # but a small MLP over a mean-pooled question summary predicts a per-example
        # additive [K, n_tokens] delta on top. No image-token x question-token dot
        # products anywhere, so this doesn't reintroduce cross-attention's O(vision x
        # question) cost -- cost is O(K x n_tokens), independent of question length.
        # Only supported with n_pool_heads==1 (multi-head pooling was already dropped
        # as a lever; combining the two axes wasn't asked for).
        self.question_conditioned = question_dim > 0
        if self.question_conditioned:
            assert n_pool_heads == 1, "question conditioning only implemented for n_pool_heads=1"
            hidden = 32
            self.question_to_pool = nn.Sequential(
                nn.Linear(question_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_compressed_tokens * n_tokens),
            )
            # zero-ish init so training starts equivalent to plain static pooling and
            # only learns to deviate -- matches this codebase's near-zero-init
            # convention elsewhere (pos_embed, glove rescale).
            self.question_to_pool[-1].weight.data.mul_(0.02)
            self.question_to_pool[-1].bias.data.zero_()

        self.add_global_token = add_global_token
        if add_global_token:
            # Mean-pooled summary of all n_tokens bridge-input positions,
            # bypassing pool_logits entirely -- cheap insurance against the
            # learned K-way compression discarding something a plain global
            # average would have kept.
            self.global_norm = nn.LayerNorm(d_model)

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, feature_map: torch.Tensor, question_summary: torch.Tensor = None) -> torch.Tensor:
        """feature_map: [B, C, H, W] -> visual tokens: [B, K (+1), d_model]

        question_summary: optional [B, question_dim] mean-pooled question embedding,
        only used when question_conditioned=True (ignored otherwise).
        """
        B, C, H, W = feature_map.shape
        n_tokens = H * W
        x = feature_map.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.in_proj(x) + self.pos_embed
        x = self.rwkv(x)  # [B, H*W, d_model]

        if self.n_pool_heads == 1:
            if self.question_conditioned and question_summary is not None:
                delta = self.question_to_pool(question_summary).view(
                    B, self.n_compressed_tokens, n_tokens
                )
                logits = self.pool_logits.unsqueeze(0) + delta  # [B, K, H*W]
                pool = torch.softmax(logits, dim=-1)
                compressed = torch.einsum("bkt,btd->bkd", pool, x)  # [B, K, d_model]
            else:
                pool = torch.softmax(self.pool_logits, dim=-1)  # [K, H*W]
                compressed = torch.einsum("kt,btd->bkd", pool, x)  # [B, K, d_model]
        else:
            pool = torch.softmax(self.multihead_pool_logits, dim=-1)  # [Heads, K, H*W]
            per_head = torch.einsum("hkt,btd->bhkd", pool, x)  # [B, Heads, K, d_model]
            concatenated = per_head.permute(0, 2, 1, 3).reshape(
                per_head.size(0), per_head.size(2), -1
            )  # [B, K, Heads*d_model]
            compressed = self.pool_merge(concatenated)  # [B, K, d_model]

        out = self.out_norm(compressed)
        if self.add_global_token:
            global_tok = self.global_norm(x.mean(dim=1, keepdim=True))  # [B, 1, d_model]
            out = torch.cat([out, global_tok], dim=1)  # [B, K+1, d_model]
        return out
