
'''Downloads the questions/annotations from the VQA dataset and converts them into a fixed top 1000 answer
classification dataset, and download the associated COCO images for this dataset.

The Dataset is then split into train/val/test, and the val and train are sampled disjointly from the train2024, 
while the test is sampled from the val2014 dataset. The test2015 dataset is not used, since there are no acommpanied answers for them.'''

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

IMAGE_URL_TMPL = "http://images.cocodataset.org/{subdir}/COCO_{subdir}_{image_id:012d}.jpg"


def load_split(raw_dir: Path, subdir: str):
    questions = json.load(
        open(raw_dir / f"v2_OpenEnded_mscoco_{subdir}_questions.json")
    )["questions"]
    annotations = json.load(
        open(raw_dir / f"v2_mscoco_{subdir}_annotations.json")
    )["annotations"]
    q_by_id = {q["question_id"]: q for q in questions}
    records = []
    for ann in annotations:
        q = q_by_id[ann["question_id"]]
        records.append(
            {
                "question_id": ann["question_id"],
                "image_id": ann["image_id"],
                "question": q["question"],
                "answer": ann["multiple_choice_answer"],
                "answer_type": ann["answer_type"],
            }
        )
    return records


def build_answer_vocab(records, vocab_size: int):
    counts = Counter(r["answer"] for r in records)
    top = [ans for ans, _ in counts.most_common(vocab_size)]
    covered = sum(counts[a] for a in top)
    coverage = covered / len(records)
    return top, counts, coverage


def download_images(image_ids_by_subdir, images_dir: Path, workers: int = 8):
    jobs = []
    for subdir, ids in image_ids_by_subdir.items():
        out_dir = images_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            dest = out_dir / f"COCO_{subdir}_{image_id:012d}.jpg"
            if not dest.exists():
                jobs.append((subdir, image_id, dest))

    session = requests.Session()

    def fetch(job):
        subdir, image_id, dest = job
        url = IMAGE_URL_TMPL.format(subdir=subdir, image_id=image_id)
        last_exc = None
        for attempt in range(6):
            try:
                resp = session.get(url, timeout=30)
                if resp.status_code in (503, 429):
                    raise requests.exceptions.HTTPError(f"{resp.status_code} for {url}")
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                return
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 20))
        raise last_exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        failed = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="images"):
            try:
                fut.result()
            except requests.exceptions.RequestException as exc:
                failed.append((futures[fut], exc))
        if failed:
            print(f"WARNING: {len(failed)} images failed to download after retries")
            for job, exc in failed[:10]:
                print(f"  {job}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="outputs/raw")
    parser.add_argument("--out-dir", default="outputs/processed")
    parser.add_argument("--images-dir", default="outputs/images")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=600)
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

    n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
    assert n_train + n_val <= len(train_pool_f), "not enough train2014 examples after filtering"
    assert n_test <= len(val_pool_f), "not enough val2014 examples after filtering"

    train_records = train_pool_f[:n_train]
    val_records = train_pool_f[n_train:n_train + n_val]
    test_records = val_pool_f[:n_test]

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
        image_ids_by_subdir = {
            "train2014": sorted({r["image_id"] for r in train_records + val_records}),
            "val2014": sorted({r["image_id"] for r in test_records}),
        }
        print(
            f"Downloading {sum(len(v) for v in image_ids_by_subdir.values())} "
            "unique COCO images referenced by the sampled subset..."
        )
        download_images(image_ids_by_subdir, Path(args.images_dir))

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
        "sampled_split_sizes": {
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
