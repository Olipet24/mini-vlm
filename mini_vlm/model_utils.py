"""Shared model construction/checkpoint-loading helpers, used by train.py
and every analysis/eval script (eval_by_type.py, eval_new_data.py,
compare_runs.py, qualitative_results.py, benchmark_efficiency.py) so they
all build a model from the exact same resolved config instead of each
re-declaring the constructor defaults independently.
"""
import io
import json
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from mini_vlm.models.baseline import LinearProjectorBaseline
from mini_vlm.models.feature_adapter import CachedFeatureAdapter
from mini_vlm.models.language_core import RWKVLanguageCore


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 * 1024)

# Mirrors the historical hardcoded constructor defaults in
# models/{baseline,language_core}.py -- every checkpoint trained before
# per-run _config.json files existed implicitly used exactly this config.
DEFAULT_CFG = dict(
    d_model=128,
    dropout=0.1,
    pool="last",
    bridge_layers=2,
    core_layers=4,
    n_compressed_tokens=8,
    baseline_layers=4,
    baseline_heads=4,
    max_question_len=16,
    token_order="vision_first",
    bridge_pool_heads=1,
    bridge_global_token=False,
    bridge_question_cond=False,
    vision_channels=576,
    vision_spatial=7,
    feature_adapter="none",
)


def build_model(name: str, vocab_size: int, num_answers: int, cfg: dict = None) -> nn.Module:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    if name == "primary":
        return RWKVLanguageCore(
            vocab_size=vocab_size,
            num_answers=num_answers,
            d_model=cfg["d_model"],
            bridge_layers=cfg["bridge_layers"],
            core_layers=cfg["core_layers"],
            n_compressed_tokens=cfg["n_compressed_tokens"],
            vision_channels=cfg["vision_channels"],
            vision_spatial=cfg["vision_spatial"],
            max_question_len=cfg["max_question_len"],
            dropout=cfg["dropout"],
            pool=cfg["pool"],
            token_order=cfg["token_order"],
            bridge_pool_heads=cfg["bridge_pool_heads"],
            bridge_global_token=cfg["bridge_global_token"],
            bridge_question_cond=cfg["bridge_question_cond"],
        )
    if name == "baseline":
        return LinearProjectorBaseline(
            vocab_size=vocab_size,
            num_answers=num_answers,
            d_model=cfg["d_model"],
            n_layer=cfg["baseline_layers"],
            n_head=cfg["baseline_heads"],
            vision_channels=cfg["vision_channels"],
            vision_spatial=cfg["vision_spatial"],
            max_question_len=cfg["max_question_len"],
            dropout=cfg["dropout"],
            pool=cfg["pool"],
        )
    raise ValueError(name)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def config_path_for(checkpoint_path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(checkpoint_path.name.replace("_checkpoint.pt", "_config.json"))


def load_model_cfg(checkpoint_path) -> dict:
    """Read the model_cfg saved alongside a checkpoint by train.py, if any.
    Older checkpoints (predating _config.json, or predating a since-added
    DEFAULT_CFG key) fall back to DEFAULT_CFG for any missing key -- callers
    read keys straight off this dict (not just through build_model(), which
    merges separately), so the merge has to happen here too."""
    candidate = config_path_for(checkpoint_path)
    if candidate.exists():
        return {**DEFAULT_CFG, **json.load(open(candidate))["model_cfg"]}
    return dict(DEFAULT_CFG)


def load_checkpoint(name: str, checkpoint_path, vocab_size: int, num_answers: int,
                     device="cpu") -> nn.Module:
    cfg = load_model_cfg(checkpoint_path)
    model = build_model(name, vocab_size, num_answers, cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def adapter_checkpoint_path_for(checkpoint_path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(checkpoint_path.name.replace("_checkpoint.pt", "_adapter_checkpoint.pt"))


def load_adapter(checkpoint_path, cfg: dict = None, device="cpu"):
    """Returns None if the given checkpoint wasn't trained with a feature
    adapter (cfg["feature_adapter"] == "none" or unset), else the loaded
    CachedFeatureAdapter -- mirrors train.py's own build-then-apply pattern
    so eval scripts reproduce the exact same forward path."""
    cfg = cfg if cfg is not None else load_model_cfg(checkpoint_path)
    mode = cfg.get("feature_adapter", "none")
    if mode == "none":
        return None
    adapter = CachedFeatureAdapter(channels=cfg["vision_channels"], mode=mode)
    state = torch.load(adapter_checkpoint_path_for(checkpoint_path), map_location=device, weights_only=True)
    adapter.load_state_dict(state)
    return adapter.to(device)
