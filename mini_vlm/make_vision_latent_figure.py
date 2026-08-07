"""Three-panel figure for the report: raw photo, the resized/cropped input the
frozen encoder actually sees, and a visualization of its latent feature map
(mean-pooled over the 576 channels, one value per spatial position) -- makes
the "offline-cached, frozen encoder" pipeline concrete rather than just prose.

Usage:
    python -m mini_vlm.make_vision_latent_figure --image report/figures/sample_bird.jpg \
        --resize 320 --out report/figures/bird_vision_latents.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from mini_vlm.data.vision_cache import build_preprocess
from mini_vlm.models.vision_encoder import build_frozen_encoder, compress_encoder_fp16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="report/figures/sample_bird.jpg")
    parser.add_argument("--resize", type=int, default=320)
    parser.add_argument("--out", default="report/figures/bird_vision_latents.png")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder = compress_encoder_fp16(build_frozen_encoder()).to(device)
    preprocess = build_preprocess(encoder.preprocess, args.resize)

    raw = ImageOps.exif_transpose(Image.open(args.image).convert("RGB"))

    resize_size = round(args.resize * 256 / 224)
    display_transform = T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(args.resize),
    ])
    processed_display = display_transform(raw)

    with torch.no_grad():
        x = preprocess(raw).unsqueeze(0).to(device).half()
        feat = encoder(x)[0].float().cpu()  # [576, H, W]
    latent = feat.mean(dim=0)  # [H, W], mean over channels

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    axes[0].imshow(raw)
    axes[0].set_title("Unprocessed", fontsize=10)
    axes[1].imshow(processed_display)
    axes[1].set_title(f"Processed ({args.resize}$\\times${args.resize})", fontsize=10)
    im = axes[2].imshow(latent, cmap="viridis")
    axes[2].set_title(f"Latent (mean of 576 ch, ${latent.shape[0]}\\times{latent.shape[1]}$)", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
