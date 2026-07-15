# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

A working PyTorch implementation exists under `mini_vlm/` (data pipeline, custom RWKV, primary
model, Transformer baseline, training/eval loop), trained end-to-end on a real VQA v2.0 subset
on CPU. See `report/technical_writeup.tex` for the authoritative file-by-file map, how-to-run
commands, done/remaining status, and progress-report rubric mapping — read that before
re-deriving any of it. `report/progress_report.tex` is the graded, page-limited course
deliverable (compile via Overleaf using `report/aps360.sty`, not locally — this machine's TeX
install is missing several base packages `environ` depends on).

Before adding tooling (formatter, linter, CI, etc.), check whether the user actually wants it — this
is an academic (APS360) course project with a hard page-limit-and-rubric report format, so scope
creep beyond what's needed for the deliverables is usually not welcome.

Environment: use `.venv` (Python 3.10 — the system default 3.14 has no PyTorch wheels yet).
`source .venv/bin/activate` before running anything under `mini_vlm/`. All training is
CPU-only by design (`torch.device("cpu")` hard-coded in `mini_vlm/train.py`); switching to GPU
is a one-line change once CUDA hardware is available.

## Project goal

Mini-VLM is a compact Vision-Language Model for Visual Question Answering (VQA) that must fit
under a **strict 100MB storage budget**, targeting edge/mobile deployment. The core idea is a
*pure RWKV* pipeline — RWKV is used both as the vision→language bridge and as the language
backbone — to avoid the quadratic memory/compute cost of standard Transformer attention.

### Architecture (as designed in the proposal)

Three-stage pipeline, described in `APS360_project_proposal-1.pdf`:

1. **Vision Encoder** — pretrained MobileNetV3, **frozen**, quantized to 8-bit via GPTQ. Since it's
   frozen, image features are meant to be extracted **once, offline, and cached** — not
   recomputed every training step. This is the key design decision for keeping training compute low.
2. **Trainable Bridge** — a custom "RWKV Spatial Bridge": processes the visual feature map as a
   linear (sequential) token track and compresses it into a small set of language-aligned tokens,
   avoiding cross-attention.
3. **Language Backbone** — a lightweight standard RWKV model that consumes the bridge's visual
   tokens plus the tokenized text question and predicts the answer.

**Baseline model** (for comparison): same frozen MobileNetV3 encoder, but with a plain Linear
Projector feeding a standard unshared Transformer instead of the RWKV bridge/core. This exists to
demonstrate that the RWKV design gives a better accuracy-to-size tradeoff, not just to have "a
baseline" — keep this comparison in mind when evaluating results.

### Data

- Dataset: **VQA v2.0** (images from COCO).
- Answers are collapsed to a **top-1000 most-common-answer classification problem** (not free-form
  text generation) — examples with rare/unique answers are dropped. This keeps the output layer
  small enough to fit the 100MB budget.
- Images are resized/normalized to **224×224** and pushed through the frozen vision encoder
  **offline**, with resulting feature tensors cached to disk for training.
- Loss: standard cross-entropy over the 1000-way answer classes.

### Constraints to respect in any implementation

- **100MB total model size is a hard budget** — this drives most architecture choices (frozen +
  compressed vision encoder, small answer vocabulary instead of open-ended generation, lightweight
  RWKV instead of attention). Currently deployed size is ~7.8MB (1.87MB FP16 vision encoder +
  5.94MB primary model state dict) — plenty of headroom, don't be afraid to scale up once on GPU,
  but recheck this number after any architecture change.
- Vision encoder is frozen; only the bridge and language core are trained.
- Prefer offline/cached feature extraction over recomputing vision features per training step.
- **Known dead end**: INT8 post-training static quantization of MobileNetV3-Small (both the plain
  torchvision model and the `torchvision.models.quantization` "quantization-ready" large variant)
  fails in this environment's PyTorch/torchvision build — the Squeeze-and-Excite block's quantized
  elementwise multiply isn't implemented end-to-end (`quantized::conv2d.new` / `empty_strided` on
  quantized tensors both error). We use FP16 compression instead (`compress_encoder_fp16` in
  `mini_vlm/models/vision_encoder.py`), which is reliable. Don't re-attempt eager-mode static INT8
  quantization on this backbone without first checking whether the PyTorch version has changed.
