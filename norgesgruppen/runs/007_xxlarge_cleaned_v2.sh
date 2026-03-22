#!/bin/bash
# 007_xxlarge_cleaned_v2.sh — Random crops + cleaned v2 labels + max dim 4000px.
#
# Combines two improvements over the current best (run 3, 0.8298):
# 1. Train on cleaned_v2 labels (bbox fixes + category reassignments),
#    evaluate on original messy labels to match test-set conditions.
# 2. Cap image max dimension at 4000px to normalize zoom level across
#    images (affects 98/248 images, up to 5712px -> 4000px).
#
# RF-DETR 2XLarge (880x880), random crops, 50/50 split.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --output-dir output-xxl-cleaned-v2-maxdim4000 \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --train-annotations data/dataset/train/_annotations.cleaned_v2.coco.json \
    --max-image-dim 4000 \
    --run-name "random-crops-xxl-cleaned-v2-maxdim4000"
