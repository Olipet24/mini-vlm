"""Side-by-side qualitative comparison of the baseline and primary models
on held-out test questions, for the progress/technical report.

Loads both trained checkpoints (outputs/results/{baseline,primary}_checkpoint.pt
by default), picks a fixed, stratified sample of test questions (spread
across answer types: yes/no, number, other) so the same examples are
reproducible run to run, and reports both models' top-3 predictions side
by side. Writes a JSON file, a plain-text table (also printed to stdout),
and an image grid figure for direct use in the report.

Usage:
    python -m mini_vlm.qualitative_results
    python -m mini_vlm.qualitative_results --results-dir outputs/results_full --n-per-type 4
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

from mini_vlm.data.dataset import VQAFeatureDataset, collate
from mini_vlm.models.baseline import LinearProjectorBaseline
from mini_vlm.models.language_core import RWKVLanguageCore
from mini_vlm.text import Tokenizer


def build_model(name: str, vocab_size: int, num_answers: int) -> torch.nn.Module:
    if name == "primary":
        return RWKVLanguageCore(vocab_size=vocab_size, num_answers=num_answers)
    if name == "baseline":
        return LinearProjectorBaseline(vocab_size=vocab_size, num_answers=num_answers)
    raise ValueError(name)


def stratified_sample(records, n_per_type: int, seed: int, types=("yes/no", "number", "other")):
    """Deterministic sample spread evenly across answer types, so the same
    question mix (some yes/no, some counting, some open-ended) shows up in
    the report every time this is re-run against a new checkpoint."""
    import random
    rng = random.Random(seed)
    by_type = {t: [r for r in records if r["answer_type"] == t] for t in types}
    sample = []
    for t in types:
        pool = by_type.get(t, [])
        rng.shuffle(pool)
        sample.extend(pool[:n_per_type])
    return sample


@torch.no_grad()
def predict_one(model, dataset, record, device):
    idx = dataset.records.index(record)
    item = dataset[idx]
    batch = collate([item])
    features = batch["features"].to(device)
    question_ids = batch["question_ids"].to(device)
    logits = model(features, question_ids)
    probs = torch.softmax(logits, dim=-1)[0]
    top3 = torch.topk(probs, k=3)
    return [
        {"answer": dataset.answer_vocab[i], "prob": round(p, 4)}
        for i, p in zip(top3.indices.tolist(), top3.values.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="outputs/processed")
    parser.add_argument("--features-dir", default="outputs/features")
    parser.add_argument("--images-dir", default="outputs/images")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--out-dir", default="outputs/results")
    parser.add_argument("--n-per-type", type=int, default=3, help="samples per answer_type (yes/no, number, other)")
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    answer_vocab = json.load(open(processed_dir / "answer_vocab.json"))
    tokenizer = Tokenizer.load(processed_dir / "tokenizer_vocab.json")
    test_ds = VQAFeatureDataset(processed_dir / "test.json", args.features_dir, tokenizer, answer_vocab)

    models = {}
    for name in ("baseline", "primary"):
        ckpt_path = results_dir / f"{name}_checkpoint.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"missing {ckpt_path} -- run `python -m mini_vlm.train --model {name}` first")
        model = build_model(name, vocab_size=len(tokenizer), num_answers=len(answer_vocab)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models[name] = model

    sample = stratified_sample(test_ds.records, args.n_per_type, args.seed)

    comparison = []
    for record in sample:
        entry = {
            "image_filename": record["image_filename"],
            "question": record["question"],
            "answer_type": record["answer_type"],
            "ground_truth_answer": record["answer"],
            "predictions": {},
        }
        for name, model in models.items():
            top3 = predict_one(model, test_ds, record, device)
            entry["predictions"][name] = {
                "top3": top3,
                "correct": top3[0]["answer"] == record["answer"],
            }
        comparison.append(entry)

    json.dump(comparison, open(out_dir / "qualitative_comparison.json", "w"), indent=2)

    col_w = 28
    header = f"{'question':<{col_w}} {'truth':<12} {'baseline (conf)':<22} {'primary (conf)':<22}"
    lines = [header, "-" * len(header)]
    for e in comparison:
        b = e["predictions"]["baseline"]["top3"][0]
        p = e["predictions"]["primary"]["top3"][0]
        b_mark = "OK" if e["predictions"]["baseline"]["correct"] else "X"
        p_mark = "OK" if e["predictions"]["primary"]["correct"] else "X"
        lines.append(
            f"{e['question'][:col_w]:<{col_w}} {e['ground_truth_answer'][:12]:<12} "
            f"{b['answer'] + ' (' + str(b['prob']) + ') ' + b_mark:<22} "
            f"{p['answer'] + ' (' + str(p['prob']) + ') ' + p_mark:<22}"
        )
    table_text = "\n".join(lines)
    print(table_text)
    (out_dir / "qualitative_comparison.txt").write_text(table_text + "\n")

    n = len(comparison)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    images_dir = Path(args.images_dir)
    for ax, e in zip(axes, comparison):
        img_path = images_dir / e["image_filename"]
        if img_path.exists():
            ax.imshow(Image.open(img_path).convert("RGB"))
        ax.axis("off")
        b = e["predictions"]["baseline"]["top3"][0]
        p = e["predictions"]["primary"]["top3"][0]
        b_color = "green" if e["predictions"]["baseline"]["correct"] else "red"
        p_color = "green" if e["predictions"]["primary"]["correct"] else "red"
        title = f"Q: {e['question']}\ntruth: {e['ground_truth_answer']}"
        ax.set_title(title, fontsize=9)
        ax.text(
            0.5, -0.08, f"baseline: {b['answer']} ({b['prob']:.2f})",
            color=b_color, fontsize=8, ha="center", va="top", transform=ax.transAxes,
        )
        ax.text(
            0.5, -0.16, f"primary: {p['answer']} ({p['prob']:.2f})",
            color=p_color, fontsize=8, ha="center", va="top", transform=ax.transAxes,
        )
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_examples.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved qualitative_comparison.json, qualitative_comparison.txt, and qualitative_examples.png to {out_dir}")


if __name__ == "__main__":
    main()
