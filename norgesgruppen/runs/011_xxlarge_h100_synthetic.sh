#!/bin/bash
# 011_xxlarge_h100_synthetic.sh — Random crops + copy-paste + synthetic data on H100 80GB.
#
# RF-DETR 2XLarge (880×880), stratified 50/50 split.
# Random crops with scale 0.5-1.5x, copy-paste augmentation,
# plus 50/50 interleaving with synthetic shelf images (200 images,
# 18K annotations, all 321 categories covered with uniform distribution).
#
# batch=64, 200 epochs, competition eval every 5 epochs.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --grad-accum 1 \
    --output-dir output-xxl-h100-synthetic \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --copy-paste \
    --synthetic-data data/synthetic_shelves \
    --synthetic-weight 0.5 \
    --run-name "random-crops-copypaste-synthetic50-xxl"
