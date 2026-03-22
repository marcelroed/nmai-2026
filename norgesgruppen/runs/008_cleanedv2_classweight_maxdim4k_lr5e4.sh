#!/bin/bash
# 008_cleanedv2_classweight_maxdim4k_lr5e4.sh — Combine best settings from runs 3-7.
#
# Combines all promising changes on top of the random-crops baseline:
#   - Cleaned v2 labels (corrected bboxes + category reassignments)
#   - Max image dim 4000px (normalize zoom across images)
#   - Per-class loss reweighting (sqrt inverse frequency)
#   - Narrower crop scale (0.7-1.3x, less extreme crops)
#   - Stronger color augmentation (brightness/contrast 0.25, sat 0.2)
#   - Lower LR (5e-4, gentler fine-tuning)
#   - 100 epochs (faster iteration)
#
# RF-DETR 2XLarge (880×880), random crops, 50% val.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 5e-4 \
    --epochs 100 \
    --batch-size 64 \
    --output-dir output-xxl-combined-best \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --train-annotations data/dataset/train/_annotations.cleaned_v2.coco.json \
    --max-image-dim 4000 \
    --class-weight-loss \
    --crop-scale-min 0.7 \
    --crop-scale-max 1.3 \
    --run-name "combined-best-cleanedv2-classweight-maxdim4k-lr5e4-scale0.7-1.3"
