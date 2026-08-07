"""Shared sequence-readout helper for the primary and baseline models.

Both models concatenate a fixed number of "prefix" tokens (the bridge's
compressed visual tokens for the primary model, the flattened vision patch
tokens for the baseline) with the question's word-token embeddings, then
reduce the resulting sequence to one vector to classify. This module is the
one place that reduction happens, so both models can be compared under the
same set of readout strategies.
"""
import torch


def pool_readout(
    x: torch.Tensor,
    question_ids: torch.Tensor,
    n_prefix_tokens: int,
    pool: str,
    pad_id: int = 0,
) -> torch.Tensor:
    """Reduce a [B, n_prefix_tokens + Tq, d] sequence to a [B, d] vector.

    'last' (the original, still-default behaviour) always reads out at the
    final *padded* position -- harmless for the bidirectional baseline
    Transformer (every position already attends to every other), but RWKV's
    per-channel exponential time-decay means the recurrent readout at a fixed
    pad position recency-weights toward however many trailing <pad> tokens
    precede it (mean question length is 6.1 words against
    max_question_len=16, i.e. often ~10 trailing pads). 'last_real' and
    'mean' both respect the question's actual (non-pad) length instead.
    """
    if pool == "last":
        return x[:, -1]
    pad_mask = question_ids != pad_id  # [B, Tq], True at real (non-pad) tokens
    if pool == "last_real":
        lengths = pad_mask.sum(dim=1).clamp(min=1)
        idx = n_prefix_tokens + lengths - 1
        return x[torch.arange(x.size(0), device=x.device), idx]
    if pool == "mean":
        prefix_mask = torch.ones(x.size(0), n_prefix_tokens, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([prefix_mask, pad_mask], dim=1).unsqueeze(-1).to(x.dtype)
        return (x * full_mask).sum(dim=1) / full_mask.sum(dim=1).clamp(min=1e-6)
    raise ValueError(f"unknown pool mode: {pool!r}")
