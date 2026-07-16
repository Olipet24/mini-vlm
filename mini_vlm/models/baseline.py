"""Baseline model: the "naive integration" control group from the
proposal. Same frozen vision encoder, but a plain Linear Projector (MLP)
feeds *all* 49 spatial patch tokens (no RWKV compression) into a
standard, unshared nn.TransformerEncoder alongside the question's word
embeddings. This is the direct point of comparison for the RWKV
primary model: quadratic self-attention over 49+Tq tokens instead of
linear-time RWKV over 8+Tq tokens.
"""
import torch
import torch.nn as nn


class LinearProjectorBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        vision_channels: int = 576,
        vision_spatial: int = 7,
        max_question_len: int = 16,
    ) -> None:
        super().__init__()
        n_patch_tokens = vision_spatial * vision_spatial
        self.visual_proj = nn.Linear(vision_channels, d_model) # MLP resampling of vision vectors
        self.token_embed = nn.Embedding(vocab_size, d_model) # learned embeddings for text
        self.pos_embed = nn.Parameter( 
            torch.randn(1, n_patch_tokens + max_question_len, d_model) * 0.02
        ) # learned vector that embeds positional knowledge into the vectors, so that transformer can have concept of word position
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True,
        ) # pre-LN (norm before each sublayer, not after) -- far more resistant to the
        # mid-training activation-magnitude blowups (loss spikes) that the default
        # post-LN layout is prone to when trained from scratch at a non-tiny LR
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer) # stacks of transformer layers/blocks
        self.final_norm = nn.LayerNorm(d_model) 
        self.classifier = nn.Linear(d_model, num_answers) # final MLP head that gives the final answer of the classification

    def forward(self, vision_features: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        B, C, H, W = vision_features.shape
        visual_tokens = self.visual_proj(vision_features.flatten(2).transpose(1, 2))  # [B, H*W, d_model]
        text_tokens = self.token_embed(question_ids)  # [B, Tq, d_model]
        x = torch.cat([visual_tokens, text_tokens], dim=1) # naive concatenation of the vision and text tokens, good for baseline
        x = x + self.pos_embed[:, : x.size(1)] # adding in the learned position embeddings 
        x = self.transformer(x)
        x = self.final_norm(x[:, -1]) # flattens the vector to apply layerNorm
        return self.classifier(x)
