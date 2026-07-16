"""Scan cached vision-feature .pt files for NaN/Inf or unreadable files.

Worth running after a training run goes non-finite, before assuming it's
purely a learning-rate issue: because loss uses mean reduction over the
batch, even one corrupted feature tensor (e.g. left behind by an
interrupted vision_cache.py run, or the flaky ~19GB COCO zip download)
poisons every batch that happens to sample it, regardless of LR.

Usage:
    python -m mini_vlm.data.check_features --features-dir outputs/features_full
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="outputs/features")
    args = parser.parse_args()

    paths = sorted(Path(args.features_dir).glob("*.pt"))
    bad = []
    for path in tqdm(paths, desc="checking features"):
        try:
            x = torch.load(path)
        except Exception as exc:
            bad.append((path.name, f"failed to load: {exc}"))
            continue
        if not torch.isfinite(x).all():
            bad.append((path.name, "contains NaN/Inf"))

    print(f"checked {len(paths)} files, {len(bad)} bad")
    for name, reason in bad[:50]:
        print(f"  {name}: {reason}")
    if len(bad) > 50:
        print(f"  ... and {len(bad) - 50} more")


if __name__ == "__main__":
    main()
