"""Primary model: RWKV Spatial Bridge + RWKV Language Core.

The bridge's K compressed visual tokens and the question's word
embeddings are concatenated into one sequence and processed by a single
shared RWKV stack (the "unified RWKV pipeline" from the proposal --
vision and language share one linear-time recurrent backbone rather
than a Transformer's quadratic self-attention). The final token's
hidden state is classified into one of the top-1000 answers.
"""
import torch
import torch.nn as nn

from mini_vlm.models.bridge import RWKVSpatialBridge
from mini_vlm.models.init import rwkv_init
from mini_vlm.models.pooling import pool_readout
from mini_vlm.models.rwkv import RWKVStack


class RWKVLanguageCore(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        d_model: int = 128,
        bridge_layers: int = 2,
        core_layers: int = 4,
        n_compressed_tokens: int = 8,
        vision_channels: int = 576,
        vision_spatial: int = 7,
        max_question_len: int = 16,
        dropout: float = 0.1,
        pool: str = "last",
        pad_id: int = 0,
        token_order: str = "vision_first",
        bridge_pool_heads: int = 1,
        bridge_global_token: bool = False,
        bridge_question_cond: bool = False,
    ) -> None:
        super().__init__()
        self.bridge_question_cond = bridge_question_cond
        self.n_compressed_tokens = n_compressed_tokens
        # bridge's actual output token count -- may exceed n_compressed_tokens
        # by 1 when bridge_global_token is on. Everything downstream that
        # sizes itself off "how many visual tokens does the bridge emit"
        # must use this, not n_compressed_tokens directly.
        self.n_prefix_tokens = n_compressed_tokens + (1 if bridge_global_token else 0)
        self.pool = pool
        self.pad_id = pad_id
        assert token_order in ("vision_first", "question_first")
        self.token_order = token_order
        self.bridge = RWKVSpatialBridge(
            in_channels=vision_channels,
            spatial_size=vision_spatial,
            d_model=d_model,
            n_layer=bridge_layers,
            n_compressed_tokens=n_compressed_tokens,
            dropout=dropout,
            n_pool_heads=bridge_pool_heads,
            add_global_token=bridge_global_token,
            question_dim=(d_model if bridge_question_cond else 0),
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_prefix_tokens + max_question_len, d_model) * 0.02
        )
        self.core = RWKVStack(n_embd=d_model, n_layer=core_layers, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_answers)

        rwkv_init(self) # initializde the embedding for the language core

    def forward(self, vision_features: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        text_tokens = self.token_embed(question_ids)
        if self.bridge_question_cond:
            mask = (question_ids != self.pad_id).unsqueeze(-1).to(text_tokens.dtype)
            question_summary = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            visual_tokens = self.bridge(vision_features, question_summary=question_summary)
        else:
            visual_tokens = self.bridge(vision_features)
        if self.token_order == "question_first":
            # question tokens first, then the K never-padded visual tokens last -- so the RWKV
            # recurrence has already ingested the full question before scanning the image, and
            # pool="last" always reads out a real vision token instead of a possibly-padded
            # question position. pool="last_real"/"mean" assume vision-first indexing and are not
            # reliable under this ordering -- only "last" is supported here.
            assert self.pool == "last", "question_first token_order only supports pool='last'"
            x = torch.cat([text_tokens, visual_tokens], dim=1)
        else:
            x = torch.cat([visual_tokens, text_tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.core(x)
        readout = pool_readout(x, question_ids, self.n_prefix_tokens, self.pool, self.pad_id)
        x = self.final_norm(readout)
        return self.classifier(self.head_drop(x))
