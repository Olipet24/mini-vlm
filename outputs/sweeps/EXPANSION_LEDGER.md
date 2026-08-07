# Expansion round ledger (Phase 0-7, this session's second sub-round)

Scratch bookkeeping, not a report file. All val_acc numbers are `checkpoint_val_acc` (the
actual restored/deployed checkpoint's val accuracy -- **not** `best_val_acc`, which was
discovered this round to be the historical max across all epochs, not necessarily what the
saved checkpoint achieves; see the `train.py` fix adding `checkpoint_val_acc`/`checkpoint_val_loss`
to metrics.json). Controls: primary `checkpoint_val_acc=0.4782` (primary_v3-equivalent config,
reproduced independently 4 times across e1/e3/e4's control arms, always 0.4782 -- good sanity
check), baseline `checkpoint_val_acc=0.4778`.

## Phase 0 -- ensemble (free, no training)

Soft-voting `primary_v3_seed{360,361,362}` on val: **0.4915**, vs best single seed 0.4782 and
baseline val mean 0.4712 (corrected). Real gain (+1.3pp over best seed). Not spent on a test
touch yet -- folded into Phase 6 as a free bonus number on the final v4 seeds.

## Phase 1 -- cheap axes (primary-only unless noted)

| Axis | Config | checkpoint_val_acc | checkpoint_val_loss | vs control | Verdict |
|---|---|---|---|---|---|
| 1a bridge-layers | 2 (control) | 0.4782 | 1.5504 | -- | control |
| | 3 | 0.4726 | 1.5543 | -0.56pp | DROP |
| | 4 | 0.4677 | 1.5431 | -1.05pp (loss better) | DROP |
| | 6 | 0.4721 | 1.5600 | -0.61pp | DROP |
| 1b reweight-other (primary) | other:1.5 | 0.4528 | 1.6255 | -1.54pp | DROP |
| | other:2.0 | 0.4466 | 1.6223 | -2.16pp | DROP |
| 1b reweight-other (baseline) | other:1.5 | 0.4639 | 1.6147 | -1.39pp (vs 0.4778) | DROP |
| | other:2.0 | 0.4593 | 1.6301 | -1.85pp | DROP |
| 1c multi-head pool | 1 (control) | 0.4782 | 1.5504 | -- | control |
| | 2 heads | 0.4728 | 1.5643 | -0.54pp | DROP |
| | 4 heads | 0.4687 | 1.5598 | -0.95pp | DROP |
| 1d global token | off (control) | 0.4782 | 1.5504 | -- | control |
| | on | 0.4778 | 1.5477 | -0.04pp (loss slightly better) | MARGINAL/neutral |
| 1e adapter (primary) | affine | 0.4754 | 1.5535 | -0.28pp | DROP |
| | conv1x1 | 0.4650 | 1.5619 | -1.32pp | DROP |
| 1e adapter (baseline) | affine | 0.4690 | 1.5561 | -0.88pp (vs 0.4778) | DROP |
| | conv1x1 | 0.4729 | 1.5351 (loss better) | -0.49pp | DROP |
| 1f glove (primary) | scale=0.02 | 0.4682 | 1.5491 | -1.00pp | DROP |
| | scale=0.1 | 0.4716 | 1.5447 | -0.66pp | DROP |
| | scale=0.5 | 0.4723 | 1.5446 | -0.59pp | DROP (best of the 3, still below control) |
| 1f glove (baseline) | scale=0.02 | 0.4598 | 1.5830 | -1.80pp (vs 0.4778) | DROP |
| | scale=0.1 | 0.4580 | 1.5991 | -1.98pp | DROP |
| | scale=0.5 | 0.4588 | 1.5859 | -1.90pp | DROP |

## Headline finding

**Every single Phase 1 lever, on both models, either hurt or was a wash.** Nothing beat the
established control. This is itself an important, honest result: hyperparameter tuning (prior
rounds) and architectural tweaks around the bridge/embeddings/features (this round) are both
exhausted at this data scale/budget -- the K=16 compression bottleneck is real and none of these
levers address it directly. The two structural levers that *do* directly address it (Phase 3
wider vocab -- doesn't touch compression but changes the problem; Phase 4 higher resolution --
gives the bridge richer raw material) are the more promising remaining directions, along with
the free ensemble result from Phase 0.

## Phase 2

Given Phase 1 found no clear primary-only architectural winner (bridge-layers/multi-head-pool
clear drops, global-token neutral), Phase 2's combination step is essentially moot for
architecture -- there's no winning lever to combine. Global-token's neutral-to-slightly-better
val_loss with negligible val_acc cost and tiny param cost (384 params) is the only candidate worth
carrying forward into Phase 3/4 trials, on the grounds that it's free and slightly helps loss.

## Phase 3 -- wider answer vocabulary (shared/mirror)

| Vocab | primary val_acc | baseline val_acc | primary vs baseline |
|---|---|---|---|
| 1000 (control) | 0.4782 | 0.4778 | primary trails by 0.04pp |
| 2000 | 0.4522 | 0.4539 | primary trails by 0.17pp |
| 3000 | 0.4478 | 0.4395 | **primary leads by 0.83pp** |

Absolute accuracy drops with wider vocab (expected: harder classification, more classes), but the
primary-vs-baseline balance shifts toward primary as vocab widens -- same direction as the
resolution finding below, weaker in magnitude. Not adopted into the final config (net accuracy
loss), but a real, consistent secondary signal worth reporting: whatever primary is missing
relative to baseline shows up less as more of the problem space (vocab) or more of the raw signal
(resolution) becomes available.

## Phase 4 -- higher input resolution (shared/mirror) -- **the clear winner of this round**

| Config | checkpoint_val_acc | checkpoint_val_loss |
|---|---|---|
| primary control (224px) | 0.4782 | 1.5504 |
| **primary 320px** | **0.4838 (+0.56pp)** | **1.5341 (better)** |
| baseline control (224px) | 0.4778 | -- |
| baseline 320px | 0.4780 (+0.02pp, flat) | 1.5200 |

Primary_320 (0.4838) now **beats** baseline_320 (0.4780) by +0.58pp -- the first clear
primary-beats-baseline result of the entire investigation (both prior hyperparameter rounds and
this round's Phase 1 architecture sweeps). Matches the diagnosed mechanism exactly: richer raw
input (100 vs 49 tokens before compression) recovers detail the K=16 bottleneck was discarding;
baseline barely moves because it was never information-starved (already saw all tokens
uncompressed even at 224px).

**3-seed confirmation (e10_resolution_confirm)**: primary_320 mean=0.4798 (std 0.0044) vs
baseline_320 mean=0.4776 (std 0.0036) -- primary wins on 2/3 individual seed pairs (360, 362) and
the 3-seed mean. Real, reproducible, if modest (+0.22pp mean) advantage. global-token on top of
320px does not help further (0.4793, worse than plain 320px's 0.4838 at seed360) -- plain
resolution alone, no extra gadgets, is the winning primary architecture. **Promoted to Phase 6 as
the final v4 config for both models** (primary: full v3 hyperparameters + vision-spatial=10;
baseline: its own tuned config + vision-spatial=10, fairness-mirrored).

## Phase 6 -- final v4 test results (single sanctioned test touch, via eval_by_type.py on the
already-trained e10_resolution/e10_resolution_confirm checkpoints -- skip-test doesn't affect
training, only whether test.json gets evaluated, so no retraining needed for this touch)

| Config | Test acc (mean, 3 seeds) | vs v3 |
|---|---|---|
| primary_v4 (320px) | **47.54% ± 0.09%** | +0.98pp (v3: 46.56%) |
| baseline_v4 (320px) | **48.01% ± 0.00%** | +0.58pp (v3: 47.43%) |
| **primary_v4 ensemble (3-seed soft-vote)** | **49.96%** | +3.40pp vs v3, **beats baseline_v4 by +1.95pp** |

Per-type (seed360): primary other 39.45%(v1)->41.46%(v3)->42.45%(v4), consistently improving.
primary yes/no now clearly beats baseline's (58.55% vs 57.87%) for the first time. number stays
close/baseline-favored throughout (28.95% vs 29.48%).

**Val vs test transfer note (same lesson as the v1 regression, now recurring)**: primary_320 won
on validation (+0.22pp mean) but *lost* to baseline_320 on the single-seed test numbers (-0.47pp)
-- gap narrowed substantially from v3 (0.87pp) to v4 (0.47pp) but didn't fully reverse on raw
single-seed test. The **ensemble** is what actually delivers a clear, decisive test win
(+1.95pp over baseline) -- a different, complementary lever (variance reduction) rather than
closing the val/test gap directly.

**Deployment note**: ensemble triples deployed size (~3x11.8MB=35.4MB primary + baseline for
comparison), still comfortably inside the 100MB budget. Report both the single-seed v4 number
(directly comparable architecture-for-architecture to baseline) and the ensemble number
(labeled as such) rather than picking one -- this is a real methodology choice worth surfacing
to the user rather than deciding unilaterally which is "the" model.

## Phase 5 -- long-T efficiency (done, see report numbers_v2.tex)

Crossover Tq=512 (fed_len~520): primary flips from slower to faster than baseline, 1.52x faster
by Tq=1000. Kernel verified numerically correct at T_MAX=1024 (max diff 6.1e-5). Long-context
accuracy probe: flat val_acc across 1x/4x/16x context multipliers (0.277/0.283/0.281,
6/24/98 words) -- no degradation at longer text, small-subset/few-epoch regime (not the final
model).

## Phase 8 -- question-conditioned bridge pooling (sweeps/e11_question_cond.json, post-report,
## triggered by user asking about co-attention) -- **the strongest single-axis result of the round**

Built on top of the v4 (320px) final primary config. Bridge's `pool_logits` [K,49->100] gets a
per-example additive delta predicted by a small MLP (hidden=32) over a mean-pooled question
embedding, instead of being static/question-blind -- a linear-time, non-attention approximation
of co-attention's "which regions matter depends on the question" signal (cost O(K x n_tokens),
no image-token x question-token dot products, so it doesn't reintroduce quadratic cost). +56,928
params (~0.2MB). Primary-only, not mirrored to baseline (same fairness rule as bridge-layers/
multi-head-pool/global-token -- baseline's Transformer already implicitly attends question<->all
tokens via self-attention, so it doesn't need or get this).

| Config | checkpoint_val_acc | checkpoint_val_loss |
|---|---|---|
| primary_320 control (e10_resolution_confirm, no q-cond) | 0.4798 mean (std 0.0044) | -- |
| **primary_320 + question-conditioned pooling** | **0.4849 mean (std 0.0018)** | 1.5214 mean |

Seeds: 360=0.4862, 361=0.4863, 362=0.4824 -- all three individually beat both the control's mean
*and* its best single seed (0.4838). +0.51pp mean over the 320px control, on top of the +0.22pp
resolution already found -- larger effect than resolution's own, and tighter across seeds (std
0.18pp vs 0.44pp) than anything else tried this round. Mechanism matches the diagnosis exactly:
static pooling was the last piece of the K=16 bottleneck architecture untouched by Phase 1 (which
tried deeper/wider/multi-head/global-token compression, all *still question-blind*) --
question-awareness in *which* positions get pooled, not just how many, was the missing lever.

**Not yet promoted to a new final config / not yet touched on test** -- this changes the
headline primary architecture after the report was already finalized at v4; promoting it means
spending a new sanctioned test touch (v5) and rewriting the report's numbers. Flagged to the user
for a go/no-go before doing either, per the same "don't unilaterally decide what's 'the' model"
principle applied to the Phase 0/6 ensemble.
