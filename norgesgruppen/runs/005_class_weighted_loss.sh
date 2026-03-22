#!/bin/bash
# 005_class_weighted_loss.sh — Random crops + per-class loss reweighting.
#
# Builds on the best run so far (random-crops-xxl-scale0.5-1.5) and adds
# per-class loss reweighting: sqrt(median_freq / class_freq), clamped to
# [0.5, 3.0]. Rare-class boxes get up to 3x the loss weight of common ones.
#
# Hypothesis: class-weighted loss on top of random crops will improve
# recall on the long tail without hurting common-class performance.
#
# RF-DETR 2XLarge (880×880), random crops, 50% val.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --output-dir output-xxl-randomcrop-classweight \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --class-weight-loss \
    --run-name "random-crops-xxl-class-weighted-loss"
