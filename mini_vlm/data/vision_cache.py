"""Run every image referenced by the processed VQA split through the
frozen, FP16-compressed MobileNetV3-Small encoder exactly once, and
cache the resulting [576, 7, 7] spatial feature map to disk. Training
scripts then only ever read cached tensors -- the vision encoder is
never touched again during training, which is the point of freezing it.

Images are loaded/preprocessed by DataLoader workers (parallel file I/O
+ resize/normalize) and run through the encoder in batches on --device,
rather than one image at a time -- needed to make this tractable over
the ~123K unique images in the full dataset (mini_vlm/data/build_dataset_full.py)
instead of just the ~4K-image CPU subset.
"""
import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mini_vlm.models.vision_encoder import (
    build_frozen_encoder,
    compress_encoder_fp16,
    state_dict_size_mb,
)


def build_preprocess(default_preprocess, resize: int):
    """default_preprocess (224px, from the bundled torchvision weights)
    unchanged when resize==224 -- byte-identical to the existing cache, no
    risk of subtly regressing it if this is ever re-run. For any other
    resize, keeps the same mean/std the frozen backbone was pretrained
    under and the same resize:crop ratio as the original 256:224 preset,
    only the crop size itself changes."""
    if resize == 224:
        return default_preprocess
    resize_size = round(resize * 256 / 224)
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(resize),
        T.ToTensor(),
        T.Normalize(mean=default_preprocess.mean, std=default_preprocess.std),
    ])


def load_all_records(processed_dir: Path):
    records = []
    for split in ("train", "val", "test"):
        records += json.load(open(processed_dir / f"{split}.json"))
    return records


class _ImageBatchDataset(Dataset):
    """One image per __getitem__ so DataLoader workers parallelize file
    I/O and CPU-side preprocessing; the encoder forward pass is then run
    batched on the target device."""

    def __init__(self, image_filenames, images_dir: Path, preprocess) -> None:
        self.image_filenames = image_filenames
        self.images_dir = images_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        image_filename = self.image_filenames[idx]
        img = Image.open(self.images_dir / image_filename).convert("RGB")
        return image_filename, self.preprocess(img)


def _collate(batch):
    filenames, tensors = zip(*batch)
    return list(filenames), torch.stack(tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="outputs/processed")
    parser.add_argument("--images-dir", default="outputs/images")
    parser.add_argument("--features-dir", default="outputs/features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="auto",
        help="'auto' picks cuda if available else cpu; or force e.g. 'cpu'/'cuda:0'",
    )
    parser.add_argument(
        "--resize", type=int, default=224,
        help="input resolution fed to the frozen encoder. 224 (default) uses the bundled "
             "torchvision preprocessing unchanged, exactly reproducing the existing cache. "
             "Any other value (e.g. 320) uses a custom resize keeping the same mean/std and "
             "resize:crop ratio, producing a *different* feature map size (320 -> 10x10) -- "
             "always point --features-dir at a new, separate directory for a non-224 resize, "
             "never mix resolutions in one cache dir.",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )

    processed_dir = Path(args.processed_dir)
    images_dir = Path(args.images_dir)
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_records(processed_dir)
    unique_images = {r["image_filename"]: r["image_id"] for r in records}
    print(f"{len(unique_images)} unique images referenced, device={device}")

    encoder = build_frozen_encoder()
    fp32_mb = state_dict_size_mb(encoder)
    encoder = compress_encoder_fp16(encoder).to(device)
    fp16_mb = state_dict_size_mb(encoder)
    print(f"Vision encoder size: {fp32_mb:.2f}MB (fp32) -> {fp16_mb:.2f}MB (fp16)")
    preprocess = build_preprocess(encoder.preprocess, args.resize)

    pending = [
        f for f in unique_images
        if not (features_dir / (Path(f).stem + ".pt")).exists()
    ]
    print(f"{len(unique_images) - len(pending)} already cached, {len(pending)} left to encode")

    loader = DataLoader(
        _ImageBatchDataset(pending, images_dir, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    observed_shape = None
    with torch.no_grad():
        for filenames, batch in tqdm(loader, desc="caching features"):
            x = batch.to(device).half()
            feats = encoder(x)  # [B, 576, H, W], fp16 -- H=W=7 at resize=224, else resize/32
            observed_shape = list(feats.shape[1:])
            for filename, feat in zip(filenames, feats):
                out_path = features_dir / (Path(filename).stem + ".pt")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat.cpu().clone(), out_path)

    if observed_shape is None:
        # nothing new to encode this run (fully resumed) -- infer shape from
        # one already-cached tensor instead of trusting the hardcoded
        # encoder.spatial_size attribute, which doesn't track --resize.
        sample = next(iter(unique_images))
        observed_shape = list(torch.load(features_dir / (Path(sample).stem + ".pt"), weights_only=True).shape)
    stats = {
        "vision_encoder_fp32_mb": round(fp32_mb, 3),
        "vision_encoder_fp16_mb": round(fp16_mb, 3),
        "num_cached_features": len(unique_images),
        "resize": args.resize,
        "feature_shape": observed_shape,
    }
    json.dump(stats, open(features_dir / "cache_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
