"""Full-scale VQA v2.0 dataset build for GPU training: same top-1000
answer-classification filtering as build_dataset.py, but keeps every
filtered example instead of sampling a small subset, and downloads the
full COCO train2014/val2014 image zips (~19GB total) rather than
fetching images one at a time -- at full scale (~388K usable train2014
examples, ~188K usable val2014 examples, ~204K unique images) a couple
of large sequential zip downloads is far more reliable than ~204K
individual HTTP GETs.

Reuses load_split/build_answer_vocab from build_dataset.py so the
top-1000 answer vocabulary logic is identical between the CPU subset
(outputs/processed) and this full run (outputs/processed_full) -- only
sampling and image acquisition differ. Writes to separate --out-dir /
--images-dir defaults so the existing CPU-subset artifacts (used in
report/) are left untouched.

Usage:
    python -m mini_vlm.data.build_dataset_full
"""
import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import requests
from tqdm import tqdm

from mini_vlm.data.build_dataset import build_answer_vocab, load_split

COCO_IMAGE_BASE = "http://images.cocodataset.org/zips"
IMAGE_ZIPS = {
    "train2014": f"{COCO_IMAGE_BASE}/train2014.zip",
    "val2014": f"{COCO_IMAGE_BASE}/val2014.zip",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def download_and_extract_images(images_dir: Path, subdirs) -> None:
    """COCO's train2014.zip/val2014.zip each extract into a top-level
    train2014/ or val2014/ folder, matching the subdir/COCO_subdir_id.jpg
    layout the rest of the pipeline (vision_cache.py, dataset.py) expects."""
    zips_dir = images_dir / "zips"
    for subdir in subdirs:
        extracted = images_dir / subdir
        if extracted.exists() and any(extracted.iterdir()):
            print(f"[skip] {subdir} images already extracted")
            continue
        url = IMAGE_ZIPS[subdir]
        zip_path = zips_dir / Path(url).name
        download_file(url, zip_path)
        print(f"Extracting {zip_path.name} (this can take a while)...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="outputs/raw")
    parser.add_argument("--out-dir", default="outputs/processed_full")
    parser.add_argument("--images-dir", default="outputs/images_full")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument(
        "--val-frac", type=float, default=0.05,
        help="fraction of the filtered train2014 pool held out as val (rest is train)",
    )
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading full train2014 and val2014 question/annotation pools...")
    train_pool = load_split(raw_dir, "train2014")
    val_pool = load_split(raw_dir, "val2014")

    print("Building top-%d answer vocabulary from the full train2014 pool..." % args.vocab_size)
    vocab, full_counts, coverage = build_answer_vocab(train_pool, args.vocab_size)
    answer_to_idx = {a: i for i, a in enumerate(vocab)}

    def filter_pool(pool):
        return [r for r in pool if r["answer"] in answer_to_idx]

    train_pool_f = filter_pool(train_pool)
    val_pool_f = filter_pool(val_pool)
    random.shuffle(train_pool_f)
    random.shuffle(val_pool_f)

    # Same source-split convention as build_dataset.py: train+val come from
    # train2014, test is entirely held out from val2014 (test2015 has no
    # published answers, so it's unused).
    n_val = int(len(train_pool_f) * args.val_frac)
    val_records = train_pool_f[:n_val]
    train_records = train_pool_f[n_val:]
    test_records = val_pool_f

    for r in train_records:
        r["subdir"] = "train2014"
        r["split"] = "train"
    for r in val_records:
        r["subdir"] = "train2014"
        r["split"] = "val"
    for r in test_records:
        r["subdir"] = "val2014"
        r["split"] = "test"

    for r in train_records + val_records + test_records:
        r["answer_idx"] = answer_to_idx[r["answer"]]
        r["image_filename"] = f"{r['subdir']}/COCO_{r['subdir']}_{r['image_id']:012d}.jpg"

    if not args.skip_download:
        print("Downloading full COCO train2014/val2014 image zips (~19GB total)...")
        download_and_extract_images(Path(args.images_dir), ["train2014", "val2014"])

    json.dump(vocab, open(out_dir / "answer_vocab.json", "w"), indent=2)
    json.dump(train_records, open(out_dir / "train.json", "w"), indent=2)
    json.dump(val_records, open(out_dir / "val.json", "w"), indent=2)
    json.dump(test_records, open(out_dir / "test.json", "w"), indent=2)

    q_lengths = [len(r["question"].split()) for r in train_records + val_records + test_records]
    answer_type_counts = Counter(r["answer_type"] for r in train_records + val_records + test_records)
    stats = {
        "full_train2014_pool_size": len(train_pool),
        "full_val2014_pool_size": len(val_pool),
        "answer_vocab_size": len(vocab),
        "top1000_answer_coverage_full_train2014": round(coverage, 4),
        "most_common_answers": full_counts.most_common(20),
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "unique_images": {
            "train+val": len({r["image_id"] for r in train_records + val_records}),
            "test": len({r["image_id"] for r in test_records}),
        },
        "question_length_words": {
            "mean": round(sum(q_lengths) / len(q_lengths), 2),
            "min": min(q_lengths),
            "max": max(q_lengths),
        },
        "answer_type_distribution": dict(answer_type_counts),
    }
    json.dump(stats, open(out_dir / "dataset_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
