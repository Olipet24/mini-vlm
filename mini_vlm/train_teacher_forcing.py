"""Small, isolated proof-of-concept: does predicting the answer as a short
autoregressive token sequence (teacher-forced during training) reduce
overfitting relative to the primary model's single-shot 1000-way
classification head, for the same underlying visual+question information?

This is explicitly scoped as a side experiment, not a replacement for the
primary model: it reuses the primary's exact bridge+core RWKV encoder
(vision_first token order, pool="last" readout) so the *encoder* overfitting
behaviour is comparable, then swaps the classification head for a tiny
GRU-cell decoder that predicts the answer as up to 3 word tokens from the
existing question tokenizer's vocabulary, teacher-forced with the true
previous token during training.

Feasibility (checked against outputs/processed_full/answer_vocab.json before
writing this): word-tokenizing the 1000 answer strings with the same
tokenizer used for questions gives length-1 answers for 862/1000, length-2
for 108, length-3 for 28, length-4 for only 2 -- so a fixed length-3 decode
covers 998/1000 answers exactly (the remaining 2 are truncated).

Never touches the test split -- this is a val-only diagnostic, not a
candidate for the final reported model.

Usage:
    python -m mini_vlm.train_teacher_forcing --processed-dir outputs/processed_full \
        --features-dir outputs/features_full --results-dir outputs/results_final \
        --run-name teacher_forcing_poc --device cuda:0
"""
import argparse
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
from mini_vlm.model_utils import git_sha, model_size_mb
from mini_vlm.models.bridge import RWKVSpatialBridge
from mini_vlm.models.rwkv import RWKVStack
from mini_vlm.text import Tokenizer

ANSWER_DECODE_LEN = 3


class TeacherForcingVQA(nn.Module):
    """Primary model's exact bridge+core encoder (vision_first, pool='last'),
    followed by a small GRU-cell decoder over ANSWER_DECODE_LEN steps instead
    of a single 1000-way classifier."""

    def __init__(self, vocab_size, d_model=128, bridge_layers=2, core_layers=4,
                 n_compressed_tokens=16, max_question_len=16, dropout=0.2):
        super().__init__()
        self.n_compressed_tokens = n_compressed_tokens
        self.bridge = RWKVSpatialBridge(
            d_model=d_model, n_layer=bridge_layers,
            n_compressed_tokens=n_compressed_tokens, dropout=dropout,
        )
        self.question_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_compressed_tokens + max_question_len, d_model) * 0.02
        )
        self.core = RWKVStack(n_embd=d_model, n_layer=core_layers, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

        # decoder: separate small embedding table for target answer tokens
        # (same vocab as questions), one GRUCell step reused across all
        # ANSWER_DECODE_LEN positions, then a linear projection back to vocab.
        self.answer_embed = nn.Embedding(vocab_size, d_model)
        self.decoder_cell = nn.GRUCell(d_model, d_model)
        self.decoder_out = nn.Linear(d_model, vocab_size)

    def encode(self, vision_features, question_ids):
        visual_tokens = self.bridge(vision_features)
        text_tokens = self.question_embed(question_ids)
        x = torch.cat([visual_tokens, text_tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.core(x)
        return self.final_norm(x[:, -1])  # pool="last"

    def forward(self, vision_features, question_ids, target_tokens=None, teacher_forcing=True):
        """target_tokens: [B, ANSWER_DECODE_LEN] or None (free-running decode).
        Returns logits [B, ANSWER_DECODE_LEN, vocab_size]."""
        h = self.encode(vision_features, question_ids)
        B = h.size(0)
        prev_token = torch.zeros(B, dtype=torch.long, device=h.device)  # pad_id=0 as <bos>
        logits_steps = []
        for t in range(ANSWER_DECODE_LEN):
            emb = self.answer_embed(prev_token)
            h = self.decoder_cell(emb, h)
            logits_t = self.decoder_out(h)
            logits_steps.append(logits_t)
            if teacher_forcing and target_tokens is not None:
                prev_token = target_tokens[:, t]
            else:
                prev_token = logits_t.argmax(dim=-1)
        return torch.stack(logits_steps, dim=1)


def build_answer_token_targets(answer_vocab, tokenizer) -> torch.Tensor:
    """[num_answers, ANSWER_DECODE_LEN] token-id targets, one row per
    answer_vocab entry, via the same word-tokenizer used for questions."""
    rows = [tokenizer.encode(ans, ANSWER_DECODE_LEN) for ans in answer_vocab]
    return torch.tensor(rows, dtype=torch.long)


def to_device_batch(batch, device):
    return batch["features"].to(device), batch["question_ids"].to(device), batch["answer_idx"].to(device)


@torch.no_grad()
def evaluate(model, loader, device, answer_token_targets, criterion):
    model.eval()
    total_loss, exact_correct, total = 0.0, 0, 0
    for batch in loader:
        features, question_ids, answer_idx = to_device_batch(batch, device)
        targets = answer_token_targets[answer_idx]  # [B, L]

        tf_logits = model(features, question_ids, targets, teacher_forcing=True)
        loss = criterion(tf_logits.reshape(-1, tf_logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item() * len(answer_idx)

        free_logits = model(features, question_ids, teacher_forcing=False)
        pred_tokens = free_logits.argmax(dim=-1)  # [B, L]
        exact_correct += (pred_tokens == targets).all(dim=-1).sum().item()
        total += len(answer_idx)
    return total_loss / total, exact_correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="outputs/processed_full")
    parser.add_argument("--features-dir", default="outputs/features_full")
    parser.add_argument("--results-dir", default="outputs/results_final")
    parser.add_argument("--run-name", default="teacher_forcing_poc")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-compressed-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"[{args.run_name}] Using device: {device}")

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    answer_vocab = json.load(open(processed_dir / "answer_vocab.json"))
    train_records = json.load(open(processed_dir / "train.json"))
    tokenizer = Tokenizer.load(processed_dir / "tokenizer_vocab.json")
    answer_token_targets = build_answer_token_targets(answer_vocab, tokenizer).to(device)

    train_ds = VQAFeatureDataset(processed_dir / "train.json", args.features_dir, tokenizer, answer_vocab)
    val_ds = VQAFeatureDataset(processed_dir / "val.json", args.features_dir, tokenizer, answer_vocab)
    if args.limit_train > 0:
        train_ds.records = train_ds.records[: args.limit_train]

    loader_kwargs = dict(collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = TeacherForcingVQA(
        vocab_size=len(tokenizer), dropout=args.dropout, n_compressed_tokens=args.n_compressed_tokens,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.run_name}] params={n_params:,} state_dict_size={model_size_mb(model):.2f}MB "
          f"decode_len={ANSWER_DECODE_LEN}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_exact_match_acc": []}
    best_val_loss, best_epoch, epochs_no_improve = None, 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{args.run_name}] epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            features, question_ids, answer_idx = to_device_batch(batch, device)
            targets = answer_token_targets[answer_idx]

            optimizer.zero_grad()
            logits = model(features, question_ids, targets, teacher_forcing=True)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(answer_idx)
            seen += len(answer_idx)
            pbar.set_postfix(loss=running_loss / seen)

        train_loss = running_loss / seen
        val_loss, val_exact_acc = evaluate(model, val_loader, device, answer_token_targets, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_exact_match_acc"].append(val_exact_acc)
        elapsed = time.time() - start
        print(f"[{args.run_name}] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_exact_match_acc={val_exact_acc:.4f} ({elapsed:.0f}s elapsed)")

        improved = best_val_loss is None or val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
        else:
            epochs_no_improve += 1
        if epoch >= args.min_epochs and epochs_no_improve >= args.early_stop_patience:
            print(f"[{args.run_name}] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    metrics = {
        "run_name": args.run_name,
        "num_params": n_params,
        "decode_len": ANSWER_DECODE_LEN,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": min(history["val_loss"]),
        "best_val_exact_match_acc": max(history["val_exact_match_acc"]),
        "final_val_loss": history["val_loss"][-1],
        "overfit_gap": history["val_loss"][-1] - min(history["val_loss"]),
        "training_time_sec": round(time.time() - start, 1),
        "history": history,
        "git_sha": git_sha(),
        "note": "val-only diagnostic; test split never touched by this script.",
    }
    json.dump(metrics, open(results_dir / f"{args.run_name}_metrics.json", "w"), indent=2)
    torch.save(model.state_dict(), results_dir / f"{args.run_name}_checkpoint.pt")

    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs_range, history["train_loss"], label="train loss")
    axes[0].plot(epochs_range, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_title(f"{args.run_name}: loss"); axes[0].legend()
    axes[1].plot(epochs_range, history["val_exact_match_acc"], label="val exact-match acc", color="green")
    axes[1].set_xlabel("epoch"); axes[1].set_title(f"{args.run_name}: val exact-match accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / f"{args.run_name}_curves.png", dpi=150)
    plt.close(fig)
    print(f"[{args.run_name}] Saved metrics, checkpoint, and curves to {results_dir}")


if __name__ == "__main__":
    main()
