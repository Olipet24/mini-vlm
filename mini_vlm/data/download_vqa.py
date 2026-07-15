"""Download VQA v2.0 question/annotation JSON archives (COCO images are
fetched separately, per-image, only for the sampled subset -- see
build_dataset.py -- since the full COCO image zips are ~19GB)."""
import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

VQA_BASE = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa"

FILES = {
    "questions_train": f"{VQA_BASE}/v2_Questions_Train_mscoco.zip",
    "annotations_train": f"{VQA_BASE}/v2_Annotations_Train_mscoco.zip",
    "questions_val": f"{VQA_BASE}/v2_Questions_Val_mscoco.zip",
    "annotations_val": f"{VQA_BASE}/v2_Annotations_Val_mscoco.zip",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))


def unzip(path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    zips_dir = out_dir / "zips"
    for name, url in FILES.items():
        dest = zips_dir / Path(url).name
        download_file(url, dest)
        unzip(dest, out_dir)
    print(f"Done. Raw VQA v2.0 questions/annotations JSON are in {out_dir}")


if __name__ == "__main__":
    main()
