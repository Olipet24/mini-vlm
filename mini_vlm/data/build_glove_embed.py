"""Build a d_model-sized GloVe-derived embedding-table init for this
project's tokenizer vocab, shared identically between primary and baseline
(train.py's --embed-init glove overwrites token_embed.weight with this,
then leaves it fully trainable -- an *initialization*, not a freeze).

Streams glove.6B.300d.txt line-by-line (never loads the full ~1GB file into
memory at once), keeps only rows for words present in the tokenizer vocab,
projects 300 -> d_model via SVD/PCA on the covered rows (deterministic, no
extra seed needed), then rescales to --init-scale (GloVe's raw vector norms
are much larger than this project's near-zero-scale init elsewhere --
rwkv_init's embedding gain, pos_embed's 0.02 -- so copying them in
unscaled would likely destabilize early training).

Usage:
    python -m mini_vlm.data.build_glove_embed \
        --glove-txt outputs/glove/glove.6B.300d.txt \
        --tokenizer-vocab outputs/processed_full/tokenizer_vocab.json \
        --d-model 128 --init-scale 0.02 \
        --out outputs/embeddings/glove_d128_processed_full.pt
"""
import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-txt", required=True)
    parser.add_argument("--tokenizer-vocab", required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--init-scale", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    word2idx = json.load(open(args.tokenizer_vocab))
    vocab_size = len(word2idx)
    print(f"Tokenizer vocab: {vocab_size} words")

    glove_dim = None
    found = {}
    with open(args.glove_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word not in word2idx:
                continue
            vec = [float(x) for x in parts[1:]]
            if glove_dim is None:
                glove_dim = len(vec)
            found[word] = vec
    assert glove_dim is not None, "no vocab words found in GloVe file -- check paths"
    coverage = len(found) / vocab_size
    print(f"GloVe dim={glove_dim}, covered {len(found)}/{vocab_size} words ({coverage:.1%})")

    covered_words = list(found.keys())
    covered_mat = torch.tensor([found[w] for w in covered_words], dtype=torch.float32)  # [n_covered, glove_dim]

    # SVD/PCA projection glove_dim -> d_model, fit on covered rows only.
    U, S, Vt = torch.linalg.svd(covered_mat - covered_mat.mean(dim=0, keepdim=True), full_matrices=False)
    proj = Vt[: args.d_model].T  # [glove_dim, d_model]
    covered_128 = covered_mat @ proj  # [n_covered, d_model]
    covered_128 = covered_128 / covered_128.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    covered_128 = covered_128 * args.init_scale

    # OOV words: same scaled-orthogonal-ish init as rwkv_init would give a
    # from-scratch nn.Embedding, so the "scratch" and "glove" conditions only
    # differ on words GloVe actually covers, not on the OOV fallback itself.
    embed = torch.randn(vocab_size, args.d_model) * args.init_scale
    for w, vec in zip(covered_words, covered_128):
        embed[word2idx[w]] = vec

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embed, out_path)
    stats = {
        "tokenizer_vocab": args.tokenizer_vocab,
        "glove_txt": args.glove_txt,
        "glove_dim": glove_dim,
        "d_model": args.d_model,
        "init_scale": args.init_scale,
        "vocab_size": vocab_size,
        "covered": len(found),
        "coverage": coverage,
        "oov_sample": [w for w in list(word2idx.keys())[:2000] if w not in found][:50],
    }
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    json.dump(stats, open(stats_path, "w"), indent=2)
    print(f"Saved {out_path} and stats json")


if __name__ == "__main__":
    main()
