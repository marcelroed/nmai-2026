#!/bin/bash
# 004_xxlarge_h100_randomcrop.sh — Training with random crops on H100 80GB.
#
# RF-DETR 2XLarge (880×880), stratified 50/50 split.
# Random crops: each epoch samples fresh 880×880 crops with random scale (0.5-1.5x).
# All images padded by 440px so boundary objects can appear in crop centers.
# Much greater data diversity than fixed patches.
#
# batch=64, 200 epochs, competition eval every 10 epochs.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --grad-accum 1 \
    --output-dir output-xxl-h100-randomcrop \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --run-name "random-crops-xxl-scale0.5-1.5"
