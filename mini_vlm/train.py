"""Shared training/eval entry point for both the RWKV primary model and
the Transformer baseline, so the two are trained and measured under
identical conditions (same data, same optimizer, same schedule).

Every architecture/regularization hyperparameter is exposed as a CLI flag
(all defaulted to reproduce the original hardcoded values with zero flags
passed) so mini_vlm/sweep.py can drive grid searches without editing code.

Usage:
    python -m mini_vlm.train --model primary  --epochs 15
    python -m mini_vlm.train --model baseline --epochs 15
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mini_vlm.data.dataset import VQAFeatureDataset, collate
from mini_vlm.model_utils import DEFAULT_CFG, build_model, git_sha, model_size_mb
from mini_vlm.models.feature_adapter import CachedFeatureAdapter
from mini_vlm.models.wkv_cuda_kernel import T_MAX as WKV_T_MAX  # single source of truth
from mini_vlm.text import UNK, Tokenizer


def to_device_batch(batch, device, ablate_vision=False):
    features = batch["features"].to(device)
    if ablate_vision:
        features = torch.zeros_like(features)
    question_ids = batch["question_ids"].to(device)
    answer_idx = batch["answer_idx"].to(device)
    return features, question_ids, answer_idx


def apply_train_augmentation(features, question_ids, feature_noise_std, word_dropout_prob, unk_id, pad_id):
    """Cheap, cache-compatible regularization applied only to training batches.
    Operates on the already-cached frozen-encoder features and tokenized
    questions -- no re-running the vision encoder, so it doesn't touch the
    offline feature-caching design. feature_noise_std adds Gaussian noise to
    the [B,576,7,7] feature tensor; word_dropout_prob randomly replaces
    non-pad question tokens with <unk>, both independently per call."""
    if feature_noise_std > 0:
        features = features + torch.randn_like(features) * feature_noise_std
    if word_dropout_prob > 0:
        question_ids = question_ids.clone()
        real_token_mask = question_ids != pad_id
        drop_mask = (torch.rand_like(question_ids, dtype=torch.float) < word_dropout_prob) & real_token_mask
        question_ids[drop_mask] = unk_id
    return features, question_ids


@torch.no_grad()
def evaluate(model, loader, device, ablate_vision=False, label_smoothing=0.0, adapter=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    for batch in loader:
        features, question_ids, answer_idx = to_device_batch(batch, device, ablate_vision)
        if adapter is not None:
            features = adapter(features)
        logits = model(features, question_ids)
        loss = criterion(logits, answer_idx)
        total_loss += loss.item() * len(answer_idx)
        correct += (logits.argmax(dim=-1) == answer_idx).sum().item()
        total += len(answer_idx)
    return total_loss / total, correct / total


def collect_qualitative(model, loader, answer_vocab, device, ablate_vision=False, n_samples=12, adapter=None):
    model.eval()
    samples = []
    with torch.no_grad():
        for batch in loader:
            features, question_ids, _ = to_device_batch(batch, device, ablate_vision)
            if adapter is not None:
                features = adapter(features)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["primary", "baseline"], required=True)
    parser.add_argument("--processed-dir", default="outputs/processed")
    parser.add_argument("--features-dir", default="outputs/features")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument(
        "--run-name", default=None,
        help="output filename stem ({run_name}_metrics.json etc.); defaults to --model, "
             "so existing single-model behaviour is unchanged. Sweeps set this to a "
             "distinct name per config to avoid collisions.",
    )
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

    # -- optimization hyperparameters (all default to the prior hardcoded behaviour) --
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="AdamW weight decay. 0.01 is torch.optim.AdamW's own default, which every "
             "prior run already used implicitly (train.py never passed weight_decay=).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # -- architecture hyperparameters (all default to models/*.py's own constructor defaults) --
    parser.add_argument("--d-model", type=int, default=DEFAULT_CFG["d_model"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--pool", choices=["last", "last_real", "mean"], default=DEFAULT_CFG["pool"])
    parser.add_argument("--bridge-layers", type=int, default=DEFAULT_CFG["bridge_layers"])
    parser.add_argument("--core-layers", type=int, default=DEFAULT_CFG["core_layers"])
    parser.add_argument("--n-compressed-tokens", type=int, default=DEFAULT_CFG["n_compressed_tokens"])
    parser.add_argument("--baseline-layers", type=int, default=DEFAULT_CFG["baseline_layers"])
    parser.add_argument("--baseline-heads", type=int, default=DEFAULT_CFG["baseline_heads"])
    parser.add_argument("--max-question-len", type=int, default=DEFAULT_CFG["max_question_len"])
    parser.add_argument(
        "--token-order", choices=["vision_first", "question_first"],
        default=DEFAULT_CFG["token_order"],
        help="primary model only: order of the bridge's K visual tokens vs. the question's word "
             "tokens in the concatenated sequence fed to the RWKV core. 'question_first' puts the "
             "never-padded visual tokens last, so pool='last' always reads a real token and the "
             "RWKV state has ingested the full question before scanning the image.",
    )
    parser.add_argument(
        "--bridge-pool-heads", type=int, default=DEFAULT_CFG["bridge_pool_heads"],
        help="primary model only: number of independent learned pooling matrices in the bridge, "
             "concatenated and merged back to d_model. 1 (default) reproduces the original single-"
             "matrix pooling exactly (checkpoint-compatible). Weights stay directly-learned "
             "nn.Parameters, not query/key dot products -- still not cross-attention.",
    )
    parser.add_argument(
        "--bridge-global-token", action="store_true",
        help="primary model only: append one extra visual token = mean-pooled summary of all "
             "bridge-input spatial positions, bypassing the learned K-way pooling entirely. "
             "Adds 1 to the T_MAX budget (n_compressed_tokens + 1 + max_question_len).",
    )
    parser.add_argument(
        "--bridge-question-cond", action="store_true",
        help="primary model only: bridge's pooling logits get a per-example additive delta "
             "predicted from a small MLP over the mean-pooled question embedding, instead of "
             "being fixed/question-blind. A linear-time, non-attention approximation of "
             "co-attention's question-relevance signal -- cost is O(K x n_tokens), independent "
             "of question length, no image-token x question-token dot products.",
    )
    parser.add_argument(
        "--vision-spatial", type=int, default=DEFAULT_CFG["vision_spatial"],
        help="side length of the frozen vision encoder's cached spatial feature map (7 for the "
             "default 224px cache, e.g. 10 for a 320px re-cache). Shared/fairness lever -- apply "
             "identically to --model primary and --model baseline when testing resolution changes.",
    )
    parser.add_argument(
        "--feature-adapter", choices=["none", "affine", "conv1x1"], default="none",
        help="small trainable transform on the cached [B,576,H,W] feature tensor, applied "
             "identically before both --model primary and --model baseline's forward pass -- "
             "a cache-compatible proxy for fine-tuning the frozen vision encoder. Shared/"
             "fairness lever, init to the identity transform so enabling it doesn't change the "
             "epoch-0 loss.",
    )
    parser.add_argument(
        "--embed-init", choices=["scratch", "glove"], default="scratch",
        help="scratch (default) reproduces the existing rwkv_init-based random init exactly. "
             "glove overwrites token_embed.weight (the question-token embedding, shared by name "
             "in both model classes) with a precomputed GloVe-derived tensor after model "
             "construction -- an initialization, the embedding stays fully trainable after. "
             "Requires --glove-embed-path. Shared/fairness lever.",
    )
    parser.add_argument(
        "--glove-embed-path", type=str, default=None,
        help="path to a .pt tensor from mini_vlm.data.build_glove_embed, required when "
             "--embed-init glove.",
    )

    # -- ablations / diagnostics --
    parser.add_argument(
        "--ablate-vision", action="store_true",
        help="zero out vision features (question-only floor: how much accuracy comes "
             "from language priors alone, independent of the image).",
    )
    parser.add_argument(
        "--limit-train", type=int, default=0,
        help="use only the first N training records (0 = all). For fast smoke tests only.",
    )
    parser.add_argument(
        "--feature-noise-std", type=float, default=0.0,
        help="Gaussian noise std added to cached vision features, training batches only "
             "(0 disables). Operates on the already-cached features, so it doesn't require "
             "re-running the frozen vision encoder.",
    )
    parser.add_argument(
        "--word-dropout-prob", type=float, default=0.0,
        help="probability of replacing each non-pad question token with <unk>, training "
             "batches only (0 disables). Standard word-dropout regularization.",
    )
    parser.add_argument(
        "--type-sample-weight", type=str, default=None,
        help="e.g. 'yes/no:1.0,other:1.5,number:1.0' -- per-answer-type training-sample "
             "weights via WeightedRandomSampler (replacement=True, num_samples=len(train_ds)). "
             "None/unset (default) keeps the existing uniform --shuffle behavior exactly. "
             "Shared/fairness-mirrored lever -- apply identically to both --model values, not "
             "primary-only.",
    )

    # -- model selection / early stopping --
    parser.add_argument(
        "--save-best", action="store_true",
        help="checkpoint the best-val-metric epoch instead of the last epoch.",
    )
    parser.add_argument("--select-metric", choices=["val_loss", "val_acc"], default="val_loss")
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="stop after this many epochs without improvement on --select-metric. 0 disables it.",
    )
    parser.add_argument("--min-epochs", type=int, default=1)

    # -- sweep hygiene --
    parser.add_argument(
        "--skip-test", action="store_true",
        help="never load test.json or evaluate on it. Mandatory for every hyperparameter-search "
             "run so model selection can never be influenced by the held-out test set -- only "
             "the final, already-chosen configuration should be run without this flag.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_name = args.run_name or args.model

    if args.model == "primary":
        core_len = args.n_compressed_tokens + (1 if args.bridge_global_token else 0) + args.max_question_len
        assert core_len <= WKV_T_MAX, (
            f"n_compressed_tokens ({args.n_compressed_tokens}) + "
            f"bridge_global_token ({1 if args.bridge_global_token else 0}) + max_question_len "
            f"({args.max_question_len}) = {core_len} exceeds the WKV CUDA kernel's "
            f"compiled T_MAX={WKV_T_MAX} (mini_vlm/models/wkv_cuda_kernel.py) -- this is the "
            f"language core's sequence length."
        )
        # The bridge's own internal RWKV stack processes vision_spatial**2 tokens
        # (before compression down to n_compressed_tokens) through the same WKV
        # kernel -- a separate, independent T_MAX budget from the core's above.
        # Missed this the first time a non-default --vision-spatial was tried
        # (320px re-cache -> 10x10=100 tokens > default T_MAX=64); MINI_VLM_WKV_T_MAX
        # must be bumped (already verified numerically correct up to 1024,
        # mini_vlm/verify_wkv_cuda.py) for any vision_spatial where this trips.
        bridge_len = args.vision_spatial ** 2
        assert bridge_len <= WKV_T_MAX, (
            f"vision_spatial ({args.vision_spatial})**2 = {bridge_len} exceeds the WKV CUDA "
            f"kernel's compiled T_MAX={WKV_T_MAX} -- this is the bridge's own internal RWKV "
            f"stack's sequence length (over the raw, uncompressed spatial tokens), independent "
            f"of the core's budget above. Set MINI_VLM_WKV_T_MAX to a value >= {bridge_len} "
            f"before running (e.g. `MINI_VLM_WKV_T_MAX=128 python -m mini_vlm.train ...`)."
        )

    model_cfg = dict(
        d_model=args.d_model,
        dropout=args.dropout,
        pool=args.pool,
        bridge_layers=args.bridge_layers,
        core_layers=args.core_layers,
        n_compressed_tokens=args.n_compressed_tokens,
        baseline_layers=args.baseline_layers,
        baseline_heads=args.baseline_heads,
        max_question_len=args.max_question_len,
        token_order=args.token_order,
        bridge_pool_heads=args.bridge_pool_heads,
        bridge_global_token=args.bridge_global_token,
        bridge_question_cond=args.bridge_question_cond,
        vision_channels=DEFAULT_CFG["vision_channels"],
        vision_spatial=args.vision_spatial,
        feature_adapter=args.feature_adapter,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"[{run_name}] Using device: {device}")

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

    train_ds = VQAFeatureDataset(
        processed_dir / "train.json", args.features_dir, tokenizer, answer_vocab,
        max_question_len=args.max_question_len,
    )
    val_ds = VQAFeatureDataset(
        processed_dir / "val.json", args.features_dir, tokenizer, answer_vocab,
        max_question_len=args.max_question_len,
    )
    if args.limit_train > 0:
        train_ds.records = train_ds.records[: args.limit_train]

    loader_kwargs = dict(
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0 and device.type == "cuda":
        # Workers are forked lazily on first iteration, by which point the
        # main process already has a CUDA context open (model.to(device)
        # below) -- forking a process with an open CUDA context is unsafe
        # and can segfault a worker partway through training. Spawning
        # instead gives each worker a clean process that never inherits it.
        loader_kwargs["multiprocessing_context"] = "spawn"

    if args.type_sample_weight:
        type_weight_map = {}
        for pair in args.type_sample_weight.split(","):
            k, v = pair.split(":")
            type_weight_map[k.strip()] = float(v)
        weights = [type_weight_map[r["answer_type"]] for r in train_ds.records]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(args.model, vocab_size=len(tokenizer), num_answers=len(answer_vocab), cfg=model_cfg).to(device)
    if args.embed_init == "glove":
        assert args.glove_embed_path, "--embed-init glove requires --glove-embed-path"
        glove_tensor = torch.load(args.glove_embed_path, map_location=device, weights_only=True)
        assert glove_tensor.shape == model.token_embed.weight.shape, (
            f"glove tensor shape {tuple(glove_tensor.shape)} != token_embed shape "
            f"{tuple(model.token_embed.weight.shape)} -- rebuild with matching vocab/d_model"
        )
        with torch.no_grad():
            model.token_embed.weight.copy_(glove_tensor)
        print(f"[{run_name}] initialized token_embed from {args.glove_embed_path} (still trainable)")
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = model_size_mb(model)
    print(f"[{run_name}] params={n_params:,} state_dict_size={size_mb:.2f}MB cfg={model_cfg}")

    adapter = None
    if args.feature_adapter != "none":
        adapter = CachedFeatureAdapter(channels=DEFAULT_CFG["vision_channels"], mode=args.feature_adapter).to(device)
    optimizer_params = list(model.parameters()) + (list(adapter.parameters()) if adapter is not None else [])
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = None
    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / args.warmup_steps, 1.0)
        )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_metric, best_epoch, best_state, best_adapter_state, epochs_no_improve = None, 0, None, None, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{run_name}] epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            features, question_ids, answer_idx = to_device_batch(batch, device, args.ablate_vision)
            features, question_ids = apply_train_augmentation(
                features, question_ids, args.feature_noise_std, args.word_dropout_prob,
                unk_id=tokenizer.word2idx[UNK], pad_id=0,
            )
            if adapter is not None:
                features = adapter(features)

            optimizer.zero_grad()
            logits = model(features, question_ids)
            loss = criterion(logits, answer_idx)
            if not torch.isfinite(loss):
                with torch.no_grad():
                    bad = torch.nonzero(~torch.isfinite(logits).all(dim=-1)).flatten().tolist()
                    print(f"\n[{run_name}] {len(bad)}/{len(answer_idx)} examples in this batch "
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
                    f"[{run_name}] loss went non-finite ({loss.item()}) at epoch {epoch}, "
                    f"{seen} examples in -- stopping now instead of wasting the rest of the run. "
                    "See per-example diagnostics above."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * len(answer_idx)
            seen += len(answer_idx)
            pbar.set_postfix(loss=running_loss / seen)

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, device, args.ablate_vision, args.label_smoothing, adapter)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        elapsed = time.time() - start
        print(
            f"[{run_name}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.0f}s elapsed)"
        )

        current = val_loss if args.select_metric == "val_loss" else val_acc
        improved = best_metric is None or (
            current < best_metric if args.select_metric == "val_loss" else current > best_metric
        )
        if improved:
            best_metric, best_epoch, epochs_no_improve = current, epoch, 0
            if args.save_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if adapter is not None:
                    best_adapter_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            epochs_no_improve += 1

        if args.early_stop_patience > 0 and epoch >= args.min_epochs and epochs_no_improve >= args.early_stop_patience:
            print(f"[{run_name}] early stopping at epoch {epoch} "
                  f"(no {args.select_metric} improvement in {args.early_stop_patience} epochs, "
                  f"best epoch {best_epoch})")
            break

    if args.save_best and best_state is not None:
        model.load_state_dict(best_state)
        if adapter is not None:
            adapter.load_state_dict(best_adapter_state)
        print(f"[{run_name}] restored best checkpoint from epoch {best_epoch} ({args.select_metric}={best_metric:.4f})")

    test_loss, test_acc, majority_acc = None, None, None
    if not args.skip_test:
        test_ds = VQAFeatureDataset(
            processed_dir / "test.json", args.features_dir, tokenizer, answer_vocab,
            max_question_len=args.max_question_len,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        test_loss, test_acc = evaluate(model, test_loader, device, args.ablate_vision, args.label_smoothing, adapter)
        print(f"[{run_name}] FINAL held-out test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        train_answer_counts = Counter(r["answer_idx"] for r in train_records)
        majority_idx, _ = train_answer_counts.most_common(1)[0]
        test_records = json.load(open(processed_dir / "test.json"))
        majority_acc = sum(1 for r in test_records if r["answer_idx"] == majority_idx) / len(test_records)

        qualitative = collect_qualitative(model, test_loader, answer_vocab, device, args.ablate_vision, adapter=adapter)
        json.dump(qualitative, open(results_dir / f"{run_name}_qualitative.json", "w"), indent=2)

    # best_val_loss/best_val_acc below are the historical best VALUES seen at
    # any epoch, which is NOT necessarily the epoch that got restored as the
    # saved checkpoint (that's best_epoch, selected on args.select_metric
    # only -- val_loss-best and val_acc-best are frequently different
    # epochs). checkpoint_val_loss/checkpoint_val_acc are what the actual
    # restored/deployed model achieves, and are what downstream comparisons
    # (sweep triage, "does X beat baseline") should use -- confirmed this
    # session that the two can differ by >1pp for val_acc specifically.
    _restored_epoch_idx = (best_epoch - 1) if best_epoch else -1
    metrics = {
        "run_name": run_name,
        "model": args.model,
        "num_params": n_params,
        "state_dict_size_mb": round(size_mb, 3),
        "epochs": args.epochs,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch if (args.save_best or args.early_stop_patience > 0) else None,
        "select_metric": args.select_metric,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "best_val_loss": min(history["val_loss"]),
        "best_val_acc": max(history["val_acc"]),
        "checkpoint_val_loss": history["val_loss"][_restored_epoch_idx],
        "checkpoint_val_acc": history["val_acc"][_restored_epoch_idx],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "majority_class_test_acc": majority_acc,
        "skip_test": args.skip_test,
        "training_time_sec": round(time.time() - start, 1),
        "history": history,
        "config": model_cfg,
    }
    json.dump(metrics, open(results_dir / f"{run_name}_metrics.json", "w"), indent=2)

    config_record = {
        "run_name": run_name,
        "model": args.model,
        "model_cfg": model_cfg,
        "train_args": vars(args),
        "git_sha": git_sha(),
    }
    json.dump(config_record, open(results_dir / f"{run_name}_config.json", "w"), indent=2)

    torch.save(model.state_dict(), results_dir / f"{run_name}_checkpoint.pt")
    if adapter is not None:
        torch.save(adapter.state_dict(), results_dir / f"{run_name}_adapter_checkpoint.pt")

    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs_range, history["train_loss"], label="train loss")
    axes[0].plot(epochs_range, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title(f"{run_name}: loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["val_acc"], label="val accuracy", color="green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{run_name}: validation accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / f"{run_name}_curves.png", dpi=150)
    plt.close(fig)
    print(f"[{run_name}] Saved metrics, config, checkpoint, and curves to {results_dir}")


if __name__ == "__main__":
    main()
