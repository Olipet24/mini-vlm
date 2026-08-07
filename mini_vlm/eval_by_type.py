"""Per-answer-type / per-slice test-set accuracy analysis for a single
trained checkpoint. Replaces the earlier untracked one-off scratch script
that produced the 57.57/42.81/29.35 (yes-no/other/number) numbers cited in
report/progress_report.tex -- this is the committed, reproducible version,
plus several additional slices that make the Discussion/Ethical
Considerations sections of the final report evidence-backed instead of
anecdotal (calibration, answer-frequency collapse, confusions).

Usage:
    python -m mini_vlm.eval_by_type --checkpoint outputs/results_final/primary_checkpoint.pt \
        --model primary --processed-dir outputs/processed_full --features-dir outputs/features_full \
        --split test --device cuda:0 --out outputs/results_final/primary_test_analysis.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_vlm.data.dataset import VQAFeatureDataset, collate
from mini_vlm.model_utils import load_adapter, load_checkpoint, load_model_cfg
from mini_vlm.text import Tokenizer

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str):
    return _WORD_RE.findall(text.lower())


def _question_prefix(question: str, n_words: int = 2) -> str:
    ws = _words(question)
    return " ".join(ws[:n_words]) if ws else ""


def _length_bucket(n_words: int) -> str:
    if n_words <= 4:
        return "1-4"
    if n_words <= 6:
        return "5-6"
    if n_words <= 8:
        return "7-8"
    return "9+"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["primary", "baseline"], required=True)
    parser.add_argument("--processed-dir", default="outputs/processed_full")
    parser.add_argument("--features-dir", default="outputs/features_full")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    processed_dir = Path(args.processed_dir)

    answer_vocab = json.load(open(processed_dir / "answer_vocab.json"))
    tokenizer = Tokenizer.load(processed_dir / "tokenizer_vocab.json")

    model_cfg = load_model_cfg(args.checkpoint)
    max_qlen = model_cfg["max_question_len"]

    ds = VQAFeatureDataset(
        processed_dir / f"{args.split}.json", args.features_dir, tokenizer, answer_vocab,
        max_question_len=max_qlen,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                         num_workers=args.num_workers)

    model = load_checkpoint(args.model, args.checkpoint, vocab_size=len(tokenizer),
                             num_answers=len(answer_vocab), device=device)
    model.eval()
    adapter = load_adapter(args.checkpoint, model_cfg, device=device)

    # train-set answer frequency, for the answer-frequency-decile slice
    train_records = json.load(open(processed_dir / "train.json"))
    train_answer_freq = Counter(r["answer_idx"] for r in train_records)
    freq_sorted = sorted(train_answer_freq.values())

    def freq_decile(answer_idx: int) -> int:
        f = train_answer_freq.get(answer_idx, 0)
        if not freq_sorted:
            return 0
        import bisect
        rank = bisect.bisect_left(freq_sorted, f) / max(len(freq_sorted), 1)
        return min(9, int(rank * 10))

    type_correct, type_total = Counter(), Counter()
    prefix_correct, prefix_total = Counter(), Counter()
    len_correct, len_total = Counter(), Counter()
    decile_correct, decile_total = Counter(), Counter()
    confusions = Counter()
    conf_sum_correct, conf_n_correct = 0.0, 0
    conf_sum_incorrect, conf_n_incorrect = 0.0, 0
    top1_correct, top5_correct, total_loss, n = 0, 0, 0.0, 0
    predicted_class_counts = Counter()
    entropy_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    idx = 0
    with torch.no_grad():
        for batch in loader:
            bs = len(batch["answer"])
            features = batch["features"].to(device)
            question_ids = batch["question_ids"].to(device)
            answer_idx = batch["answer_idx"].to(device)
            if adapter is not None:
                features = adapter(features)

            logits = model(features, question_ids)
            total_loss += criterion(logits, answer_idx).item()
            probs = torch.softmax(logits, dim=-1)
            top5 = torch.topk(probs, k=5, dim=-1)
            pred1 = top5.indices[:, 0]
            max_prob = top5.values[:, 0]
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)

            hits1 = (pred1 == answer_idx)
            hits5 = (top5.indices == answer_idx.unsqueeze(1)).any(dim=1)
            top1_correct += hits1.sum().item()
            top5_correct += hits5.sum().item()
            n += bs
            entropy_sum += entropy.sum().item()

            recs = ds.records[idx: idx + bs]
            for i in range(bs):
                r = recs[i]
                truth_idx = r["answer_idx"]
                pred_idx = pred1[i].item()
                correct = bool(hits1[i].item())
                atype = r.get("answer_type", "unknown")
                nwords = len(_words(r["question"]))

                type_total[atype] += 1
                type_correct[atype] += correct

                prefix = _question_prefix(r["question"])
                prefix_total[prefix] += 1
                prefix_correct[prefix] += correct

                lb = _length_bucket(nwords)
                len_total[lb] += 1
                len_correct[lb] += correct

                dec = freq_decile(truth_idx)
                decile_total[dec] += 1
                decile_correct[dec] += correct

                predicted_class_counts[pred_idx] += 1
                p = max_prob[i].item()
                if correct:
                    conf_sum_correct += p
                    conf_n_correct += 1
                else:
                    conf_sum_incorrect += p
                    conf_n_incorrect += 1
                    confusions[(r["answer"], answer_vocab[pred_idx])] += 1
            idx += bs

    top_prefixes = sorted(
        ((p, c, prefix_correct[p] / c) for p, c in prefix_total.items() if c >= 20),
        key=lambda t: -t[1],
    )[:10]

    result = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "split": args.split,
        "n": n,
        "model_cfg": model_cfg,
        "overall": {
            "top1_acc": top1_correct / n,
            "top5_acc": top5_correct / n,
            "loss": total_loss / n,
            "mean_prediction_entropy": entropy_sum / n,
            "n_distinct_predicted_classes": len(predicted_class_counts),
            "n_answer_classes": len(answer_vocab),
        },
        "by_answer_type": {
            t: {"acc": type_correct[t] / type_total[t], "n": type_total[t]}
            for t in sorted(type_total)
        },
        "by_question_prefix_top10": [
            {"prefix": p, "n": c, "acc": acc} for p, c, acc in top_prefixes
        ],
        "by_question_length": {
            lb: {"acc": len_correct[lb] / len_total[lb], "n": len_total[lb]}
            for lb in sorted(len_total)
        },
        "by_answer_frequency_decile": {
            str(d): {"acc": decile_correct[d] / decile_total[d], "n": decile_total[d]}
            for d in sorted(decile_total)
        },
        "confidence": {
            "mean_max_prob_correct": (conf_sum_correct / conf_n_correct) if conf_n_correct else None,
            "mean_max_prob_incorrect": (conf_sum_incorrect / conf_n_incorrect) if conf_n_incorrect else None,
        },
        "top_confusions": [
            {"truth": t, "predicted": p, "count": c}
            for (t, p), c in confusions.most_common(15)
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)

    print(f"[{args.model}] {args.split} n={n} top1_acc={result['overall']['top1_acc']:.4f} "
          f"top5_acc={result['overall']['top5_acc']:.4f} loss={result['overall']['loss']:.4f}")
    for t, d in result["by_answer_type"].items():
        print(f"  {t:<10} acc={d['acc']:.4f} n={d['n']}")
    print(f"  mean_max_prob correct={result['confidence']['mean_max_prob_correct']:.4f} "
          f"incorrect={result['confidence']['mean_max_prob_incorrect']:.4f}")
    print(f"Saved full analysis to {out_path}")


if __name__ == "__main__":
    main()
