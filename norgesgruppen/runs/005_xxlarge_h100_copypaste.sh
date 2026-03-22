#!/bin/bash
# 005_xxlarge_h100_copypaste.sh — Random crops + copy-paste augmentation on H100 80GB.
#
# RF-DETR 2XLarge (880×880), stratified 50/50 split.
# Random crops with scale 0.5-1.5x, plus copy-paste augmentation that:
#   - Extracts all object instances into an instance bank
#   - Pastes 2-3 instances per crop, biased toward confused-pair counterparts
#   - 30% of image selections biased toward confused-pair images
#
# batch=64, 200 epochs, competition eval every 5 epochs.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --grad-accum 1 \
    --output-dir output-xxl-h100-copypaste \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --copy-paste \
    --run-name "random-crops-copypaste-xxl"
