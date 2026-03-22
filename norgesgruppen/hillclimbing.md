# Hillclimbing Tracker

Tracking experiments to improve the competition score (0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5).

wandb project: https://wandb.ai/ankile/norgesgruppen-rfdetr

## Current Best: `random-crops-xxl-cleaned-labels` — 0.8410 @ epoch 19 (run 6, still training)

## Results (best score so far, epoch in parentheses)

| # | Run name | Det mAP | Cls mAP | Combined | Epoch | Status | Notes |
|---|----------|--------:|--------:|---------:|------:|--------|-------|
| 1 | `baseline-xxl-patched` | 0.917 | 0.607 | 0.824 | 9 | Failed | Fixed patches. Peaked early, declined after. |
| 2 | `no-patches-xxl-resize-only` | 0.769 | 0.405 | 0.660 | 89 | Failed | Resize only. **Discarded** |
| 3 | `random-crops-xxl-scale0.5-1.5` | 0.910 | 0.653 | 0.833 | 29 | Running | Random crops baseline |
| 4 | `oversample-balanced-confused-pairs` | — | — | — | — | Crashed | |
| 5 | [`class-weighted-loss`](https://wandb.ai/ankile/norgesgruppen-rfdetr/runs/mj4jb35x) | 0.921 | 0.636 | **0.836** | 14 | Running | = Run 3 + class-weighted loss |
| 6 | [`cleaned-labels`](https://wandb.ai/ankile/norgesgruppen-rfdetr/runs/ecwkmzgg) | 0.919 | 0.660 | **0.841** | 19 | Running | = Run 3 + cleaned labels. **Best** |
| 7 | [`copypaste`](https://wandb.ai/ankile/norgesgruppen-rfdetr/runs/97kl09au) | 0.920 | 0.634 | 0.834 | 9 | Running | = Run 3 + copy-paste aug |
| 8 | [`cleaned-v2-maxdim4000`](https://wandb.ai/ankile/norgesgruppen-rfdetr/runs/oegb66bu) | 0.925 | 0.628 | 0.836 | 9 | Running | = Run 3 + cleaned_v2 + maxdim 4000 |
| 9 | [`combined-best`](https://wandb.ai/ankile/norgesgruppen-rfdetr/runs/7sd6x948) | 0.914 | 0.495 | 0.788 | 4 | Running | = Run 8 + classweight + lr=5e-4 + scale 0.7-1.3. Very early |
| 10 | `submission-banker` | — | — | — | — | Running | Full-train, no val. Based on run 8 config |
| 11 | `submission-banker-copypaste-classweight` | — | — | — | — | Running | Full-train + copy-paste + class-weighted loss |

## Epoch-Aligned Comparison

Comparing at **epoch 9** (the latest epoch where most runs have data):

| Run | Det mAP | Cls mAP | Combined | Delta vs Run 3 |
|-----|--------:|--------:|---------:|:--------------:|
| 3 (baseline random crops) | 0.920 | 0.619 | 0.830 | — |
| 5 (+ class-weighted loss) | 0.921 | 0.620 | 0.831 | +0.001 |
| 6 (+ cleaned labels) | 0.918 | 0.628 | 0.831 | +0.001 |
| 7 (+ copy-paste) | 0.920 | 0.634 | 0.834 | +0.004 |
| 8 (+ cleaned_v2 + maxdim4k) | 0.925 | 0.628 | 0.836 | +0.006 |

At **epoch 19** (runs 3, 5, 6 only):

| Run | Det mAP | Cls mAP | Combined | Delta vs Run 3 |
|-----|--------:|--------:|---------:|:--------------:|
| 3 (baseline random crops) | 0.913 | 0.644 | 0.832 | — |
| 5 (+ class-weighted loss) | 0.914 | 0.643 | 0.833 | +0.001 |
| 6 (+ cleaned labels) | 0.919 | 0.660 | **0.841** | **+0.009** |

All runs: RF-DETR 2XLarge, lr=1e-3 (except run 9: 5e-4), batch_size=64, 50% val, seed=42.

**Excluded images**: `img_00295.jpg` — Dense knekkebrød shelf with many incorrect annotations.

## Observations

### Run 1: fixed patches — peaked early, declined
- Best at epoch 9 (0.824), then declined to 0.814 by epoch 39
- Fixed patches memorize quickly, then overfit

### Run 2: resize only — DISCARDED
- Plateaued at 0.660 after 100+ epochs. Too much information lost.

### Run 3: random crops — solid baseline (0.833)
- Peaked at epoch 29 (0.833), then flat/slight decline
- Classification climbs steadily (0.619 → 0.653)

### Run 5: class-weighted loss — marginal improvement
- Tracks ~0.001 ahead of baseline at same epochs
- Not a significant difference — the reweighting is roughly neutral

### Run 6: cleaned labels — CLEAR BEST (+0.009 at epoch 19)
- The only run that clearly separates from baseline at matched epochs
- Classification mAP at 0.660 vs baseline 0.644 — cleaned labels help classification most
- Still climbing at epoch 19, may improve further

### Run 7: copy-paste — promising early (+0.004 at epoch 9)
- Slightly ahead at epoch 9 (0.834 vs 0.830), needs more epochs to confirm

### Run 8: cleaned_v2 + maxdim4000 — best detection
- Highest det mAP (0.925) at epoch 9
- Classification still catching up (0.628)
- The maxdim normalization seems to help detection

### Run 9: combined best — slow start (epoch 4 only)
- Lower LR (5e-4) means slower warmup, 0.788 at epoch 4
- Need more data to judge

### Takeaways
1. **Cleaned labels are the biggest proven win** — +0.009 at matched epochs
2. **Maxdim 4000 boosts detection** — highest det mAP across all runs
3. **Class-weighted loss is roughly neutral** — ~+0.001
4. **Copy-paste looks promising early** — needs more epochs
5. **Random crops >> fixed patches** — confirmed (fixed patches overfit)
