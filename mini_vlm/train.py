"""Shared training/eval entry point for both the RWKV primary model and
the Transformer baseline, so the two are trained and measured under
identical conditions (same data, same optimizer, same schedule).

Usage:
    python -m mini_vlm.train --model primary  --epochs 15
    python -m mini_vlm.train --model baseline --epochs 15
"""
import argparse
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mini_vlm.data.dataset import VQAFeatureDataset, collate
from mini_vlm.models.baseline import LinearProjectorBaseline
from mini_vlm.models.language_core import RWKVLanguageCore
from mini_vlm.text import Tokenizer


def build_model(name: str, vocab_size: int, num_answers: int) -> nn.Module:
    if name == "primary":
        return RWKVLanguageCore(vocab_size=vocab_size, num_answers=num_answers)
    if name == "baseline":
        return LinearProjectorBaseline(vocab_size=vocab_size, num_answers=num_answers)
    raise ValueError(name)


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 * 1024)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        features = batch["features"].to(device)
        question_ids = batch["question_ids"].to(device)
        answer_idx = batch["answer_idx"].to(device)
        logits = model(features, question_ids)
        loss = criterion(logits, answer_idx)
        total_loss += loss.item() * len(answer_idx)
        correct += (logits.argmax(dim=-1) == answer_idx).sum().item()
        total += len(answer_idx)
    return total_loss / total, correct / total


def collect_qualitative(model, loader, git, answer_vocab, device, n_samples=12):
    model.eval()
    samples = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            question_ids = batch["question_ids"].to(device)
            logits = model(features, question_ids)
            probs = torch.softmax(logits, dim=-1)
            top3 = torch.topk(probs, k=3, dim=-1)
            for i in range(len(batch["question"])):
                samples.append({
                    "image_filename": batch["image_filename"][i],
                    "question": batch["question"][i],
                    "ground_truth_answer": batch["answer"][i],
                    "top3_predictions": [
                        {"answer": answer_vocab[idx], "prob": round(prob, 4)}
                        for idx, prob in zip(top3.indices[i].tolist(), top3.values[i].tolist())
                    ],
                    "correct": answer_vocab[top3.indices[i, 0].item()] == batch["answer"][i],
                })
                if len(samples) >= n_samples:
                    return samples
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["primary", "baseline"], required=True)
    parser.add_argument("--processed-dir", default="outputs/processed")
    parser.add_argument("--features-dir", default="outputs/features")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument(
        "--device", default="auto",
        help="'auto' picks cuda if available else cpu; or force e.g. 'cpu'/'cuda:0'",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes; >0 only helps once data loading is the bottleneck (i.e. on GPU)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="linear LR warmup over this many optimizer steps, then constant at --lr. "
             "0 (default) disables it, matching prior runs exactly. The baseline's "
             "post-LN nn.TransformerEncoder is prone to NaN early in training without "
             "warmup; the RWKV primary model doesn't need it but it's harmless there too.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    answer_vocab = json.load(open(processed_dir / "answer_vocab.json"))
    train_records = json.load(open(processed_dir / "train.json"))

    vocab_path = processed_dir / "tokenizer_vocab.json"
    if vocab_path.exists():
        tokenizer = Tokenizer.load(vocab_path)
    else:
        tokenizer = Tokenizer.build([r["question"] for r in train_records])
        tokenizer.save(vocab_path)

    train_ds = VQAFeatureDataset(processed_dir / "train.json", args.features_dir, tokenizer, answer_vocab)
    val_ds = VQAFeatureDataset(processed_dir / "val.json", args.features_dir, tokenizer, answer_vocab)
    test_ds = VQAFeatureDataset(processed_dir / "test.json", args.features_dir, tokenizer, answer_vocab)

    loader_kwargs = dict(
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(args.model, vocab_size=len(tokenizer), num_answers=len(answer_vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = model_size_mb(model)
    print(f"[{args.model}] params={n_params:,} state_dict_size={size_mb:.2f}MB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / args.warmup_steps, 1.0)
        )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{args.model}] epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            features = batch["features"].to(device)
            question_ids = batch["question_ids"].to(device)
            answer_idx = batch["answer_idx"].to(device)

            optimizer.zero_grad()
            logits = model(features, question_ids)
            loss = criterion(logits, answer_idx)
            if not torch.isfinite(loss):
                with torch.no_grad():
                    bad = torch.nonzero(~torch.isfinite(logits).all(dim=-1)).flatten().tolist()
                    print(f"\n[{args.model}] {len(bad)}/{len(answer_idx)} examples in this batch "
                          f"have non-finite logits:")
                    for i in bad[:5]:
                        feat = features[i]
                        print(
                            f"  image={batch['image_filename'][i]!r} question={batch['question'][i]!r} "
                            f"answer={batch['answer'][i]!r} feat[min={feat.min().item():.4g} "
                            f"max={feat.max().item():.4g} mean={feat.mean().item():.4g} "
                            f"has_nan={torch.isnan(feat).any().item()} has_inf={torch.isinf(feat).any().item()}]"
                        )
                raise SystemExit(
                    f"[{args.model}] loss went non-finite ({loss.item()}) at epoch {epoch}, "
                    f"{seen} examples in -- stopping now instead of wasting the rest of the run. "
                    "See per-example diagnostics above."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * len(answer_idx)
            seen += len(answer_idx)
            pbar.set_postfix(loss=running_loss / seen)

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        elapsed = time.time() - start
        print(
            f"[{args.model}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.0f}s elapsed)"
        )

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[{args.model}] FINAL held-out test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    # majority-class baseline for reference
    from collections import Counter
    train_answer_counts = Counter(r["answer_idx"] for r in train_records)
    majority_idx, majority_count = train_answer_counts.most_common(1)[0]
    test_records = json.load(open(processed_dir / "test.json"))
    majority_acc = sum(1 for r in test_records if r["answer_idx"] == majority_idx) / len(test_records)

    metrics = {
        "model": args.model,
        "num_params": n_params,
        "state_dict_size_mb": round(size_mb, 3),
        "epochs": args.epochs,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "majority_class_test_acc": majority_acc,
        "training_time_sec": round(time.time() - start, 1),
        "history": history,
    }
    json.dump(metrics, open(results_dir / f"{args.model}_metrics.json", "w"), indent=2)

    qualitative = collect_qualitative(model, test_loader, tokenizer, answer_vocab, device)
    json.dump(qualitative, open(results_dir / f"{args.model}_qualitative.json", "w"), indent=2)

    torch.save(model.state_dict(), results_dir / f"{args.model}_checkpoint.pt")

    epochs_range = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs_range, history["train_loss"], label="train loss")
    axes[0].plot(epochs_range, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title(f"{args.model}: loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["val_acc"], label="val accuracy", color="green")
    axes[1].axhline(majority_acc, color="gray", linestyle="--", label="majority-class baseline")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{args.model}: validation accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / f"{args.model}_curves.png", dpi=150)
    print(f"Saved metrics, qualitative samples, checkpoint, and curves to {results_dir}")


if __name__ == "__main__":
    main()
