"""Evaluate both trained checkpoints on newly collected, never-seen-during-
tuning data (images the user photographed themselves + hand-written
questions/answers) -- the infrastructure for the final report's "Evaluate
model on new data" rubric criterion (10 points, the single largest line item
on top of the progress report).

`questions.csv` columns: image_filename,question,answer,answer_type
  - image_filename must exist under --images-dir
  - answer must be exactly one of the 1000 strings in answer_vocab.json --
    this is a closed-set classifier, not open-ended generation
  - answer_type must be one of {yes/no, number, other}

Always run --validate-only first (and again after adding more rows) --
it catches every mistake below before wasting a labelling session:

    python -m mini_vlm.eval_new_data --validate-only \
        --images-dir data_new/images --questions data_new/questions.csv \
        --processed-dir outputs/processed_full

Then the real evaluation, once the checkpoints in --results-dir exist:

    python -m mini_vlm.eval_new_data \
        --images-dir data_new/images --questions data_new/questions.csv \
        --processed-dir outputs/processed_full --results-dir outputs/results_final \
        --models primary,baseline --out-dir outputs/results_new_data --device cuda:0
"""
import argparse
import csv
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9]+")  # mirrors mini_vlm.text's tokenization regex exactly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from mini_vlm.data.vision_cache import build_preprocess
from mini_vlm.model_utils import load_checkpoint, load_model_cfg
from mini_vlm.models.vision_encoder import build_frozen_encoder, compress_encoder_fp16
from mini_vlm.text import Tokenizer

VALID_ANSWER_TYPES = {"yes/no", "number", "other"}


def read_questions_csv(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    missing_cols = {"image_filename", "question", "answer", "answer_type"} - set(reader.fieldnames or [])
    if missing_cols:
        raise SystemExit(f"{path} is missing required column(s): {sorted(missing_cols)}")
    return rows


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% confidence interval for a binomial proportion --
    the right interval to use at n~100, where a naive p +/- 1.96*sqrt(p(1-p)/n)
    normal approximation gets unreliable near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def validate(rows, images_dir: Path, answer_vocab: list, tokenizer_words: set):
    answer_set = set(answer_vocab)
    errors, warnings = [], []
    seen = set()
    type_counts = Counter()

    for i, r in enumerate(rows, start=2):  # row 1 is the header
        img_path = images_dir / r["image_filename"]
        if not img_path.exists():
            errors.append(f"line {i}: image not found: {img_path}")
        else:
            try:
                Image.open(img_path).verify()
            except Exception as exc:
                errors.append(f"line {i}: unreadable image {img_path}: {exc}")

        answer = r["answer"]
        if answer not in answer_set:
            near = difflib.get_close_matches(answer, answer_vocab, n=5, cutoff=0.4)
            near_str = f" nearest in-vocab matches: {near}" if near else " (no close matches found)"
            errors.append(f"line {i}: answer {answer!r} is not in the top-1000 answer vocabulary.{near_str}")

        if r["answer_type"] not in VALID_ANSWER_TYPES:
            errors.append(f"line {i}: answer_type {r['answer_type']!r} not in {sorted(VALID_ANSWER_TYPES)}")
        else:
            type_counts[r["answer_type"]] += 1

        key = (r["image_filename"], r["question"].strip().lower())
        if key in seen:
            errors.append(f"line {i}: duplicate (image_filename, question) pair: {key}")
        seen.add(key)

        q_words = _WORD_RE.findall(r["question"].lower())
        oov = [w for w in q_words if w not in tokenizer_words]
        if oov:
            warnings.append(f"line {i}: question has {len(oov)}/{len(q_words)} word(s) not in the "
                             f"training tokenizer vocabulary (will map to <unk>): {oov}")

    return errors, warnings, type_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--questions", required=True, help="path to questions.csv")
    parser.add_argument("--processed-dir", default="outputs/processed_full")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--results-dir", default="outputs/results_final")
    parser.add_argument("--models", default="primary,baseline")
    parser.add_argument("--out-dir", default="outputs/results_new_data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-grid", type=int, default=8)
    parser.add_argument("--resize", type=int, default=224,
                         help="must match the vision_spatial the target checkpoints were trained "
                              "at (320 for the final v5/v4 models, vision_spatial=10) -- the "
                              "encoder's bundled preprocess() defaults to 224px/7x7 and will "
                              "produce a shape mismatch against a 320px-trained bridge otherwise.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    processed_dir = Path(args.processed_dir)
    rows = read_questions_csv(Path(args.questions))
    print(f"Loaded {len(rows)} rows from {args.questions}")

    answer_vocab = json.load(open(processed_dir / "answer_vocab.json"))
    tokenizer = Tokenizer.load(processed_dir / "tokenizer_vocab.json")
    tokenizer_words = set(tokenizer.word2idx.keys())

    errors, warnings, type_counts = validate(rows, images_dir, answer_vocab, tokenizer_words)

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings[:30]:
            print(f"  [warn] {w}")
        if len(warnings) > 30:
            print(f"  ... and {len(warnings) - 30} more")

    if errors:
        print(f"\n{len(errors)} error(s) -- fix these before evaluating:")
        for e in errors:
            print(f"  [error] {e}")
        raise SystemExit(f"\nvalidation FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")

    print(f"\nvalidation OK: {len(rows)} rows, 0 errors, {len(warnings)} warning(s)")
    print(f"answer_type mix: {dict(type_counts)}")
    if len(rows) < 60:
        print(f"[note] n={len(rows)} is below the ~60-question floor for a defensible 95% CI "
              f"on accuracy (see mini_vlm/eval_new_data.py docstring) -- keep collecting.")

    if args.validate_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = args.models.split(",")

    # -- vision features, computed in-process (no cache needed at this scale) --
    encoder = compress_encoder_fp16(build_frozen_encoder()).to(device)
    preprocess = build_preprocess(encoder.preprocess, args.resize)
    features_by_file = {}
    with torch.no_grad():
        for r in rows:
            fname = r["image_filename"]
            if fname in features_by_file:
                continue
            img = ImageOps.exif_transpose(Image.open(images_dir / fname).convert("RGB"))
            x = preprocess(img).unsqueeze(0).to(device).half()
            feat = encoder(x)[0].float().cpu()  # [576, 7, 7]
            features_by_file[fname] = feat

    # -- per-model inference --
    predictions = defaultdict(dict)  # model_name -> row_index -> pred info
    models = {}
    for name in model_names:
        ckpt_path = results_dir / f"{name}_checkpoint.pt"
        model_cfg = load_model_cfg(ckpt_path)
        model = load_checkpoint(name, ckpt_path, vocab_size=len(tokenizer),
                                 num_answers=len(answer_vocab), device=device)
        model.eval()
        models[name] = (model, model_cfg["max_question_len"])

    all_items = []
    with torch.no_grad():
        for idx, r in enumerate(rows):
            feat = features_by_file[r["image_filename"]].unsqueeze(0).to(device)
            item = {
                "image_filename": r["image_filename"],
                "question": r["question"],
                "answer": r["answer"],
                "answer_type": r["answer_type"],
                "predictions": {},
            }
            for name, (model, max_qlen) in models.items():
                q_ids = torch.tensor([tokenizer.encode(r["question"], max_qlen)], dtype=torch.long, device=device)
                logits = model(feat, q_ids)
                probs = torch.softmax(logits, dim=-1)[0]
                top3 = torch.topk(probs, k=3)
                pred = {
                    "top3": [
                        {"answer": answer_vocab[i], "prob": round(p, 4)}
                        for i, p in zip(top3.indices.tolist(), top3.values.tolist())
                    ],
                    "correct": answer_vocab[top3.indices[0].item()] == r["answer"],
                }
                item["predictions"][name] = pred
            all_items.append(item)

    json.dump(all_items, open(out_dir / "new_data_predictions.json", "w"), indent=2)

    # -- summary with Wilson 95% CIs --
    summary = {"n": len(rows), "per_model": {}}
    train_records = json.load(open(processed_dir / "train.json")) if (processed_dir / "train.json").exists() else []
    if train_records:
        maj_idx, maj_count = Counter(r["answer_idx"] for r in train_records).most_common(1)[0]
        majority_answer = answer_vocab[maj_idx]
        majority_correct = sum(1 for it in all_items if it["answer"] == majority_answer)
        lo, hi = wilson_ci(majority_correct, len(rows))
        summary["majority_class_floor"] = {"answer": majority_answer, "acc": majority_correct / len(rows),
                                            "ci95": [lo, hi]}

    for name in model_names:
        n_correct = sum(1 for it in all_items if it["predictions"][name]["correct"])
        lo, hi = wilson_ci(n_correct, len(rows))
        by_type = defaultdict(lambda: [0, 0])
        confidences = []
        for it in all_items:
            t = it["answer_type"]
            by_type[t][1] += 1
            correct = it["predictions"][name]["correct"]
            by_type[t][0] += int(correct)
            confidences.append(it["predictions"][name]["top3"][0]["prob"])
        summary["per_model"][name] = {
            "acc": n_correct / len(rows),
            "ci95": [lo, hi],
            "n_correct": n_correct,
            "n": len(rows),
            "mean_top1_confidence": sum(confidences) / len(confidences),
            "by_answer_type": {
                t: {"acc": c / n, "n": n, "ci95": list(wilson_ci(c, n))}
                for t, (c, n) in by_type.items()
            },
        }

    json.dump(summary, open(out_dir / "new_data_summary.json", "w"), indent=2)
    with open(out_dir / "new_data_summary.md", "w") as f:
        f.write(f"# New-data evaluation (n={len(rows)})\n\n")
        if "majority_class_floor" in summary:
            mf = summary["majority_class_floor"]
            f.write(f"Majority-class floor (train-set mode answer {mf['answer']!r}): "
                    f"{mf['acc']:.3f} (95% CI [{mf['ci95'][0]:.3f}, {mf['ci95'][1]:.3f}])\n\n")
        f.write("| model | acc | 95% CI | yes/no | other | number |\n|---|---|---|---|---|---|\n")
        for name, s in summary["per_model"].items():
            bt = s["by_answer_type"]
            row = lambda t: f"{bt[t]['acc']:.3f} (n={bt[t]['n']})" if t in bt else "n/a"
            f.write(f"| {name} | {s['acc']:.3f} | [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                    f"{row('yes/no')} | {row('other')} | {row('number')} |\n")

    print(f"\n=== New-data results (n={len(rows)}) ===")
    for name, s in summary["per_model"].items():
        print(f"  {name}: acc={s['acc']:.4f} 95%CI=[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}]")

    # -- compact image grid --
    grid_items = all_items[: args.max_grid]
    if grid_items:
        n = len(grid_items)
        cols = min(4, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4.5 * rows_n))
        axes = axes.flatten() if n > 1 else [axes]
        for ax, item in zip(axes, grid_items):
            img = Image.open(images_dir / item["image_filename"]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            lines = [f"Q: {item['question']}", f"truth: {item['answer']}"]
            for name in model_names:
                p = item["predictions"][name]["top3"][0]
                mark = "yes" if item["predictions"][name]["correct"] else "no"
                lines.append(f"{name}: {p['answer']} ({p['prob']:.2f}) [{mark}]")
            ax.set_title("\n".join(lines), fontsize=8)
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "new_data_grid.png", dpi=130)
        plt.close(fig)

    print(f"Saved predictions, summary, and grid to {out_dir}")


if __name__ == "__main__":
    main()
