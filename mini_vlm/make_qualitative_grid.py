"""Generates the compact qualitative-results figure for the final report:
one held-out test question per category -- both models correct, only
primary correct, only baseline correct, both wrong -- rendered as a small
image grid with each model's top prediction and confidence overlaid.
Replaces the older mini_vlm/qualitative_results.py (which predates
model_utils-based config-aware checkpoint loading and can't load a
checkpoint trained with non-default architecture hyperparameters, e.g.
the final primary model's n_compressed_tokens=16).

Usage:
    python -m mini_vlm.make_qualitative_grid \
        --primary-checkpoint outputs/results_final/primary_seed360_checkpoint.pt \
        --baseline-checkpoint outputs/results_final/baseline_seed360_checkpoint.pt \
        --processed-dir outputs/processed_full --features-dir outputs/features_full \
        --images-dir outputs/images_full --out report/figures/qualitative_grid_final.png
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from mini_vlm.data.dataset import VQAFeatureDataset, collate
from mini_vlm.model_utils import load_checkpoint
from mini_vlm.text import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--processed-dir", default="outputs/processed_full")
    parser.add_argument("--features-dir", default="outputs/features_full")
    parser.add_argument("--images-dir", default="outputs/images_full")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=360)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    processed_dir = Path(args.processed_dir)

    answer_vocab = json.load(open(processed_dir / "answer_vocab.json"))
    tokenizer = Tokenizer.load(processed_dir / "tokenizer_vocab.json")
    test_ds = VQAFeatureDataset(processed_dir / "test.json", args.features_dir, tokenizer, answer_vocab)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=4)

    primary = load_checkpoint("primary", args.primary_checkpoint, len(tokenizer), len(answer_vocab), device=device)
    baseline = load_checkpoint("baseline", args.baseline_checkpoint, len(tokenizer), len(answer_vocab), device=device)
    primary.eval()
    baseline.eval()

    buckets = {"both_correct": [], "primary_only": [], "baseline_only": [], "both_wrong": []}
    idx = 0
    with torch.no_grad():
        for batch in loader:
            bs = len(batch["answer"])
            features = batch["features"].to(device)
            question_ids = batch["question_ids"].to(device)
            p_logits = primary(features, question_ids)
            b_logits = baseline(features, question_ids)
            p_probs = torch.softmax(p_logits, dim=-1)
            b_probs = torch.softmax(b_logits, dim=-1)
            p_top1 = p_probs.argmax(dim=-1)
            b_top1 = b_probs.argmax(dim=-1)
            recs = test_ds.records[idx: idx + bs]
            for i, r in enumerate(recs):
                truth_idx = r["answer_idx"]
                p_correct = p_top1[i].item() == truth_idx
                b_correct = b_top1[i].item() == truth_idx
                entry = {
                    "record": r,
                    "primary_pred": answer_vocab[p_top1[i].item()],
                    "primary_prob": p_probs[i, p_top1[i]].item(),
                    "baseline_pred": answer_vocab[b_top1[i].item()],
                    "baseline_prob": b_probs[i, b_top1[i]].item(),
                    "primary_correct": p_correct,
                    "baseline_correct": b_correct,
                }
                if p_correct and b_correct:
                    buckets["both_correct"].append(entry)
                elif p_correct and not b_correct:
                    buckets["primary_only"].append(entry)
                elif b_correct and not p_correct:
                    buckets["baseline_only"].append(entry)
                else:
                    buckets["both_wrong"].append(entry)
            idx += bs

    rng = random.Random(args.seed)
    images_dir = Path(args.images_dir)
    chosen = []
    for key in ("both_correct", "primary_only", "baseline_only", "both_wrong"):
        pool = [e for e in buckets[key] if (images_dir / e["record"]["image_filename"]).exists()]
        rng.shuffle(pool)
        if pool:
            chosen.append((key, pool[0]))
        print(f"{key}: {len(buckets[key])} candidates in test set")

    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4.6))
    if len(chosen) == 1:
        axes = [axes]
    for ax, (key, e) in zip(axes, chosen):
        r = e["record"]
        img = Image.open(images_dir / r["image_filename"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Q: {r['question']}\ntruth: {r['answer']}", fontsize=9)
        b_color = "green" if e["baseline_correct"] else "red"
        p_color = "green" if e["primary_correct"] else "red"
        ax.text(0.5, -0.08, f"baseline: {e['baseline_pred']} ({e['baseline_prob']:.2f})",
                color=b_color, fontsize=8, ha="center", va="top", transform=ax.transAxes)
        ax.text(0.5, -0.16, f"primary: {e['primary_pred']} ({e['primary_prob']:.2f})",
                color=p_color, fontsize=8, ha="center", va="top", transform=ax.transAxes)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # also dump the chosen examples for reference/citation
    json.dump(
        [{"category": k, **{kk: vv for kk, vv in e.items() if kk != "record"}, "question": e["record"]["question"],
          "answer": e["record"]["answer"], "answer_type": e["record"]["answer_type"]}
         for k, e in chosen],
        open(out_path.with_suffix(".json"), "w"), indent=2,
    )


if __name__ == "__main__":
    main()
