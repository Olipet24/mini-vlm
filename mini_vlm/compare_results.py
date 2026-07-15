"""Combine baseline and primary model metrics (written by train.py) into a
single side-by-side comparison plot and summary table, e.g. Figure
comparison_curves in report/technical_writeup.tex -- run this after
training both models with the same --results-dir.

Usage:
    python -m mini_vlm.compare_results --results-dir outputs/results_full
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"baseline": "tab:orange", "primary": "tab:blue"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics = {}
    for name in ("baseline", "primary"):
        path = results_dir / f"{name}_metrics.json"
        if not path.exists():
            raise SystemExit(f"missing {path} -- run `python -m mini_vlm.train --model {name}` first")
        metrics[name] = json.load(open(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, m in metrics.items():
        epochs = range(1, len(m["history"]["val_loss"]) + 1)
        axes[0].plot(epochs, m["history"]["val_loss"], label=name, color=COLORS[name])
        axes[1].plot(epochs, m["history"]["val_acc"], label=name, color=COLORS[name])

    majority_acc = next(iter(metrics.values()))["majority_class_test_acc"]
    axes[1].axhline(majority_acc, color="gray", linestyle="--", label="majority-class floor")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation loss")
    axes[0].set_title("Validation loss")
    axes[0].legend()

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title("Validation accuracy")
    axes[1].legend()

    fig.tight_layout()
    out_path = results_dir / "comparison_curves.png"
    fig.savefig(out_path, dpi=150)

    header = f"{'model':<10} {'params':>12} {'size_mb':>10} {'test_acc':>10} {'test_loss':>10}"
    print(header)
    for name, m in metrics.items():
        print(f"{name:<10} {m['num_params']:>12,} {m['state_dict_size_mb']:>10.2f} "
              f"{m['test_acc']:>10.4f} {m['test_loss']:>10.4f}")
    print(f"majority-class floor: {majority_acc:.4f}")
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
