"""Generalized N-config sweep aggregator. compare_results.py stays as the
simple baseline-vs-primary-only comparison it always was (technical_writeup.tex
references it by name); this is its sibling for comparing an arbitrary number
of sweep runs against each other -- the tool that turns a directory of
mini_vlm.sweep output into a leaderboard, a config-diff table, and an
overfitting-gap column that answers "did this hyperparameter actually help".

Usage:
    python -m mini_vlm.compare_runs --results-dir outputs/sweeps/s1_primary_reg \
        --sort-by best_val_loss --top 15 --plot --out-prefix outputs/sweeps/s1_primary_reg/summary
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# config keys that meaningfully vary a run -- excluded from "what varies"
# detection since they're identifiers/bookkeeping, not hyperparameters.
_NON_CONFIG_COLUMNS = {"run_name", "model"}


def load_runs(results_dirs):
    runs = []
    for d in results_dirs:
        d = Path(d)
        for mpath in sorted(d.glob("*_metrics.json")):
            metrics = json.load(open(mpath))
            history = metrics.get("history", {})
            val_loss_hist = history.get("val_loss", [])
            val_acc_hist = history.get("val_acc", [])
            best_val_loss = metrics.get("best_val_loss", min(val_loss_hist) if val_loss_hist else None)
            best_val_acc = metrics.get("best_val_acc", max(val_acc_hist) if val_acc_hist else None)
            best_epoch = metrics.get("best_epoch")
            if best_epoch is None and val_loss_hist:
                best_epoch = val_loss_hist.index(best_val_loss) + 1
            final_val_loss = metrics.get("final_val_loss", val_loss_hist[-1] if val_loss_hist else None)
            runs.append({
                "results_dir": str(d),
                "run_name": metrics.get("run_name", mpath.stem.replace("_metrics", "")),
                "model": metrics.get("model"),
                "config": metrics.get("config", {}),
                "num_params": metrics.get("num_params"),
                "state_dict_size_mb": metrics.get("state_dict_size_mb"),
                "training_time_sec": metrics.get("training_time_sec"),
                "epochs_run": metrics.get("epochs_run", len(val_loss_hist)),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "final_val_loss": final_val_loss,
                "overfit_gap": (final_val_loss - best_val_loss) if (final_val_loss is not None and best_val_loss is not None) else None,
                "history": history,
                "test_acc": metrics.get("test_acc"),
                "test_loss": metrics.get("test_loss"),
            })
    return runs


def varying_config_keys(runs):
    keys = set()
    for r in runs:
        keys.update(r["config"].keys())
    varying = []
    for k in sorted(keys):
        values = {json.dumps(r["config"].get(k)) for r in runs}
        if len(values) > 1:
            varying.append(k)
    return varying


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", action="append", required=True,
                         help="repeatable; one or more sweep output dirs to glob *_metrics.json from")
    parser.add_argument("--sort-by", choices=["best_val_loss", "best_val_acc"], default="best_val_loss")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no *_metrics.json found under {args.results_dir}")

    reverse = args.sort_by == "best_val_acc"
    runs.sort(key=lambda r: (r[args.sort_by] is None, r[args.sort_by]), reverse=reverse)
    top_runs = runs[: args.top]

    varying = varying_config_keys(runs)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # -- CSV + Markdown table --
    columns = ["run_name", "model"] + varying + [
        "num_params", "state_dict_size_mb", "best_epoch", "best_val_loss",
        "best_val_acc", "final_val_loss", "overfit_gap", "training_time_sec",
    ]
    with open(f"{out_prefix}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in top_runs:
            row = [r["run_name"], r["model"]] + [r["config"].get(k) for k in varying] + [
                r["num_params"], r["state_dict_size_mb"], r["best_epoch"],
                round(r["best_val_loss"], 4) if r["best_val_loss"] is not None else None,
                round(r["best_val_acc"], 4) if r["best_val_acc"] is not None else None,
                round(r["final_val_loss"], 4) if r["final_val_loss"] is not None else None,
                round(r["overfit_gap"], 4) if r["overfit_gap"] is not None else None,
                r["training_time_sec"],
            ]
            writer.writerow(row)

    with open(f"{out_prefix}.md", "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for r in top_runs:
            row = [r["run_name"], r["model"]] + [str(r["config"].get(k)) for k in varying] + [
                str(r["num_params"]), f"{r['state_dict_size_mb']:.2f}" if r["state_dict_size_mb"] else "",
                str(r["best_epoch"]),
                f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "",
                f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None else "",
                f"{r['final_val_loss']:.4f}" if r["final_val_loss"] is not None else "",
                f"{r['overfit_gap']:.4f}" if r["overfit_gap"] is not None else "",
                f"{r['training_time_sec']:.0f}" if r["training_time_sec"] else "",
            ]
            f.write("| " + " | ".join(row) + " |\n")

    # -- leaderboard printout --
    print(f"Loaded {len(runs)} runs from {args.results_dir}; varying config keys: {varying}")
    print(f"Top {len(top_runs)} by {args.sort_by}:")
    header = f"{'run_name':<28}{'model':<10}" + "".join(f"{k:<16}" for k in varying) + f"{'best_val_loss':<15}{'best_val_acc':<14}{'overfit_gap':<12}"
    print(header)
    for r in top_runs:
        cfg_str = "".join(f"{str(r['config'].get(k)):<16}" for k in varying)
        bvl = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "n/a"
        bva = f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None else "n/a"
        gap = f"{r['overfit_gap']:.4f}" if r["overfit_gap"] is not None else "n/a"
        print(f"{r['run_name']:<28}{str(r['model']):<10}{cfg_str}{bvl:<15}{bva:<14}{gap:<12}")

    # -- overlay curves --
    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for r in top_runs:
            hist = r["history"]
            if not hist.get("val_loss"):
                continue
            epochs = range(1, len(hist["val_loss"]) + 1)
            axes[0].plot(epochs, hist["val_loss"], label=r["run_name"], alpha=0.8)
            axes[1].plot(epochs, hist["val_acc"], label=r["run_name"], alpha=0.8)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("val loss")
        axes[0].set_title("Validation loss")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val accuracy")
        axes[1].set_title("Validation accuracy")
        if len(top_runs) <= 15:
            axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_curves.png", dpi=150)
        plt.close(fig)
        print(f"Saved overlay curves to {out_prefix}_curves.png")

    print(f"Saved {out_prefix}.csv and {out_prefix}.md")


if __name__ == "__main__":
    main()
