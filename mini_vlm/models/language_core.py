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
    ) -> None:
        super().__init__()
        self.bridge = RWKVSpatialBridge(
            in_channels=vision_channels,
            spatial_size=vision_spatial,
            d_model=d_model,
            n_layer=bridge_layers,
            n_compressed_tokens=n_compressed_tokens,
            dropout=dropout,
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_compressed_tokens + max_question_len, d_model) * 0.02
        )
        self.core = RWKVStack(n_embd=d_model, n_layer=core_layers, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_answers)

        rwkv_init(self) # initializde the embedding for the language core

    def forward(self, vision_features: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.bridge(vision_features)
        text_tokens = self.token_embed(question_ids)
        x = torch.cat([visual_tokens, text_tokens], dim=1) # would be interesting to see how we can implement this and improve using cross attention
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.core(x)
        x = self.final_norm(x[:, -1])  # last token's hidden state
        return self.classifier(self.head_drop(x))
