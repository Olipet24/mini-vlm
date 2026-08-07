"""Measures forward-pass latency/throughput/peak-memory for both models as
the question length grows, to put a real number on the project's actual
value proposition -- linear-time RWKV vs. quadratic-time self-attention --
since overall test accuracy alone is nearly tied between primary and
baseline and doesn't make that case on its own.

For each question length Tq swept, the baseline's Transformer attends over
49 (fixed patch tokens) + Tq positions; the primary model's RWKV core
processes n_compressed_tokens (8 by default) + Tq positions. Both grow with
Tq, but at different asymptotic rates -- that's what this script measures,
not the absolute numbers (random weights, dummy inputs; this is a compute
benchmark, not an accuracy one, so it doesn't need a trained checkpoint).

Usage:
    python -m mini_vlm.benchmark_efficiency --results-dir outputs/results_final \
        --device cuda:0 --seq-lens 8,16,32,48 --out outputs/results_final/efficiency.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mini_vlm.model_utils import DEFAULT_CFG, build_model, load_model_cfg
from mini_vlm.models.wkv_cuda_kernel import T_MAX as WKV_T_MAX  # single source of truth


def _cfg_for(model_name: str, results_dir: Path) -> dict:
    candidate = results_dir / f"{model_name}_config.json"
    if candidate.exists():
        return load_model_cfg(results_dir / f"{model_name}_checkpoint.pt")
    return dict(DEFAULT_CFG)


def _prefix_len(model_name: str, cfg: dict) -> int:
    # baseline keeps all vision_spatial**2 raw patch tokens uncompressed (49 at the default
    # 224px cache, 100 at a 320px re-cache) -- was hardcoded to 49, silently wrong for any
    # non-default --vision-spatial config.
    return cfg["n_compressed_tokens"] if model_name == "primary" else cfg["vision_spatial"] ** 2


@torch.no_grad()
def _bench_one(model_name, cfg, seq_len, vocab_size, num_answers, batch_size, device, n_warmup, n_iters):
    cfg = {**cfg, "max_question_len": seq_len}
    model = build_model(model_name, vocab_size=vocab_size, num_answers=num_answers, cfg=cfg).to(device)
    model.eval()

    vs = cfg["vision_spatial"]
    vision_features = torch.randn(batch_size, cfg["vision_channels"], vs, vs, device=device)
    question_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for _ in range(n_warmup):
        model(vision_features, question_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model(vision_features, question_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None
    median_ms = statistics.median(latencies)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "seq_len": seq_len,
        "fed_len": _prefix_len(model_name, cfg) + seq_len,
        "median_latency_ms": median_ms,
        "p10_latency_ms": statistics.quantiles(latencies, n=10)[0] if len(latencies) >= 10 else min(latencies),
        "p90_latency_ms": statistics.quantiles(latencies, n=10)[-1] if len(latencies) >= 10 else max(latencies),
        "throughput_samples_per_sec": batch_size / (median_ms / 1000.0),
        "peak_memory_mb": peak_mem_mb,
        "num_params": n_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs/results_final",
                         help="looks for {model}_config.json here to match the final trained "
                              "architecture; falls back to DEFAULT_CFG if absent.")
    parser.add_argument("--processed-dir", default="outputs/processed_full",
                         help="only used to size the embedding tables realistically; optional.")
    parser.add_argument("--seq-lens", default="8,16,32,48",
                         help="comma-separated question lengths (Tq) to benchmark at.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iters", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="outputs/results_final/efficiency.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    vocab_size, num_answers = 12267, 1000
    pdir = Path(args.processed_dir)
    if (pdir / "tokenizer_vocab.json").exists():
        vocab_size = len(json.load(open(pdir / "tokenizer_vocab.json")))
    if (pdir / "answer_vocab.json").exists():
        num_answers = len(json.load(open(pdir / "answer_vocab.json")))

    results_dir = Path(args.results_dir)
    results = {}
    for model_name in ("primary", "baseline"):
        cfg = _cfg_for(model_name, results_dir)
        points = []
        for seq_len in seq_lens:
            if model_name == "primary" and cfg["n_compressed_tokens"] + seq_len > WKV_T_MAX:
                print(f"[skip] primary seq_len={seq_len}: n_compressed_tokens+seq_len="
                      f"{cfg['n_compressed_tokens'] + seq_len} exceeds WKV T_MAX={WKV_T_MAX}")
                continue
            point = _bench_one(model_name, cfg, seq_len, vocab_size, num_answers,
                                args.batch_size, device, args.n_warmup, args.n_iters)
            points.append(point)
            print(f"[{model_name}] Tq={seq_len:>3} fed_len={point['fed_len']:>3} "
                  f"median={point['median_latency_ms']:.3f}ms "
                  f"throughput={point['throughput_samples_per_sec']:.1f}/s "
                  f"peak_mem={point['peak_memory_mb']}")
        results[model_name] = {"cfg": cfg, "points": points}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "device": str(device),
        "batch_size": args.batch_size,
        "vocab_size": vocab_size,
        "num_answers": num_answers,
        "results": results,
    }, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model_name, style in (("primary", "o-"), ("baseline", "s-")):
        pts = results[model_name]["points"]
        if not pts:
            continue
        ax.plot([p["fed_len"] for p in pts], [p["median_latency_ms"] for p in pts], style, label=model_name)
    ax.set_xlabel("total tokens fed to core (log scale)")
    ax.set_ylabel("median forward latency (ms, log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Forward latency vs. sequence length (batch={args.batch_size})")
    ax.legend()
    fig.tight_layout()
    plot_path = out_path.with_name(out_path.stem + "_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
