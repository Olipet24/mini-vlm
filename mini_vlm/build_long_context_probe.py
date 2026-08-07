"""Small, isolated auxiliary experiment (not a real VQA claim): build
synthetic long-context question variants by prepending other real
questions' text as filler before the real target question, so we can train
the *existing, unmodified* primary architecture at question lengths well
beyond VQA's actual ~6-word average and see (a) whether accuracy holds up
and (b) pair that with mini_vlm.benchmark_efficiency's long-T latency
crossover finding (primary becomes faster than baseline around Tq~512,
report/final_report_v2.tex Discussion).

This is explicitly a controlled probe, not a claim about real long-question
VQA performance -- filler-question context is not naturalistic long text.
Same image/answer as the real target throughout; only the question text
grows. Reuses the existing outputs/processed_full/{answer_vocab,
tokenizer_vocab}.json unchanged (vocab doesn't depend on this), and the
existing feature cache unchanged (images unchanged) -- only train.json/
val.json get a synthetic "question" field, so mini_vlm.train can be run
against the output directory completely unmodified.

Usage:
    python -m mini_vlm.build_long_context_probe --processed-dir outputs/processed_full \
        --multiplier 16 --n-train 3000 --n-val 1000 \
        --out-dir outputs/processed_full_longctx_16x --seed 360
"""
import argparse
import json
import random
from pathlib import Path


def build_split(records, multiplier, n_sample, rng):
    pool = records
    sampled = rng.sample(records, min(n_sample, len(records)))
    out = []
    for r in sampled:
        r = dict(r)
        if multiplier > 1:
            # Target question FIRST, filler after -- Tokenizer.encode truncates
            # from the end at max_question_len, so this order guarantees any
            # truncation only ever drops filler, never the real target question.
            fillers = rng.sample(pool, multiplier - 1)
            filler_text = " ".join(f["question"] for f in fillers)
            r["question"] = f"{r['question']} {filler_text}"
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="outputs/processed_full")
    parser.add_argument("--multiplier", type=int, required=True,
                         help="1 = control (real questions unchanged), 4/16 = prepend that many "
                              "extra real questions' text as filler before the real target question.")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=360)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = json.load(open(processed_dir / "train.json"))
    val_records = json.load(open(processed_dir / "val.json"))

    train_out = build_split(train_records, args.multiplier, args.n_train, rng)
    val_out = build_split(val_records, args.multiplier, args.n_val, rng)

    json.dump(train_out, open(out_dir / "train.json", "w"))
    json.dump(val_out, open(out_dir / "val.json", "w"))

    # answer_vocab/tokenizer_vocab are unchanged by this (same answers, same
    # word vocab -- filler text reuses real in-vocab questions) -- copy so
    # mini_vlm.train can point --processed-dir at out_dir directly with zero
    # other changes.
    for fname in ("answer_vocab.json", "tokenizer_vocab.json"):
        json.dump(json.load(open(processed_dir / fname)), open(out_dir / fname, "w"))

    mean_words = sum(len(r["question"].split()) for r in train_out) / len(train_out)
    stats = {
        "multiplier": args.multiplier,
        "n_train": len(train_out),
        "n_val": len(val_out),
        "mean_question_words": round(mean_words, 2),
        "seed": args.seed,
    }
    json.dump(stats, open(out_dir / "longctx_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
