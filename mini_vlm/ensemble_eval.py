"""Free (no training) check: does soft-voting several already-trained
checkpoints of the same model beat the best single seed?

Val-only by design -- this is a diagnostic to decide whether ensembling is
worth folding into the final round, not a candidate for a sanctioned test
touch on its own (see Discussion in report/final_report_v2.tex for the
test-set-touch discipline this project follows).

Usage:
    python -m mini_vlm.ensemble_eval --model primary \
        --checkpoints outputs/results_final/primary_v3_seed360_checkpoint.pt,\
outputs/results_final/primary_v3_seed361_checkpoint.pt,\
outputs/results_final/primary_v3_seed362_checkpoint.pt \
        --processed-dir outputs/processed_full --features-dir outputs/features_full \
        --split val --device cuda:0 --out outputs/results_final/ensemble_v3_val_analysis.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_vlm.data.dataset import VQAFeatureDataset, collate
from mini_vlm.model_utils import load_checkpoint
from mini_vlm.text import Tokenizer


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["primary", "baseline"], required=True)
    parser.add_argument("--checkpoints", required=True, help="comma-separated checkpoint paths")
    parser.add_argument("--processed-dir", default="outputs/processed_full")
    parser.add_argument("--features-dir", default="outputs/features_full")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    processed_dir = Path(args.processed_dir)

    answer_vocab = json.load(open(processed_dir / "answer_vocab.json"))
    tokenizer = Tokenizer.load(processed_dir / "tokenizer_vocab.json")

    ckpt_paths = [p.strip() for p in args.checkpoints.split(",")]
    models = [
        load_checkpoint(args.model, p, vocab_size=len(tokenizer), num_answers=len(answer_vocab), device=device).eval()
        for p in ckpt_paths
    ]
    print(f"Loaded {len(models)} checkpoints: {ckpt_paths}")

    ds = VQAFeatureDataset(processed_dir / f"{args.split}.json", args.features_dir, tokenizer, answer_vocab)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                         num_workers=args.num_workers)

    per_seed_correct = [0] * len(models)
    ensemble_soft_correct = 0
    ensemble_hard_correct = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        question_ids = batch["question_ids"].to(device)
        answer_idx = batch["answer_idx"].to(device)

        all_probs = []
        all_preds = []
        for i, model in enumerate(models):
            logits = model(features, question_ids)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)
            preds = probs.argmax(dim=-1)
            all_preds.append(preds)
            per_seed_correct[i] += (preds == answer_idx).sum().item()

        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        soft_pred = avg_probs.argmax(dim=-1)
        ensemble_soft_correct += (soft_pred == answer_idx).sum().item()

        stacked_preds = torch.stack(all_preds, dim=0)  # [n_models, B]
        for b in range(stacked_preds.size(1)):
            votes = Counter(stacked_preds[:, b].tolist())
            top_count = max(votes.values())
            tied = [k for k, v in votes.items() if v == top_count]
            if len(tied) == 1:
                hard_pred = tied[0]
            else:
                # tie-break by highest-confidence single model among the tied classes
                best_i, best_p, best_c = 0, -1.0, tied[0]
                for i in range(len(models)):
                    c = stacked_preds[i, b].item()
                    if c in tied and all_probs[i][b, c].item() > best_p:
                        best_i, best_p, best_c = i, all_probs[i][b, c].item(), c
                hard_pred = best_c
            if hard_pred == answer_idx[b].item():
                ensemble_hard_correct += 1

        total += len(answer_idx)

    per_seed_acc = [c / total for c in per_seed_correct]
    result = {
        "model": args.model,
        "split": args.split,
        "checkpoints": ckpt_paths,
        "n": total,
        "per_seed_acc": per_seed_acc,
        "best_single_seed_acc": max(per_seed_acc),
        "ensemble_soft_vote_acc": ensemble_soft_correct / total,
        "ensemble_hard_vote_acc": ensemble_hard_correct / total,
        "soft_vote_gain_over_best_seed": ensemble_soft_correct / total - max(per_seed_acc),
    }
    print(json.dumps(result, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
