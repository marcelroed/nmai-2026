#!/bin/bash
# 010_submission_banker_copypaste.sh — Full-train submission with copy-paste augmentation.
#
# Same as 009 (cleaned_v2 labels, maxdim 4000, random crops, full train) plus:
#   - Copy-paste augmentation (instance bank, confused-pair biased)
#   - Per-class loss reweighting (sqrt inverse frequency)
#   - Stronger color augmentation (via updated defaults in augmentation.py)
#
# After training, export with:
#   uv run prepare-submission --checkpoint output-submission-banker-copypaste/checkpoint_best_total.pth
#
# IMPORTANT: run.py MAX_IMAGE_DIM must be set to 4000 to match training.
#
# RF-DETR 2XLarge (880×880), random crops, 100% train, 200 epochs.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --output-dir output-submission-banker-copypaste \
    --full-train \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --train-annotations data/dataset/train/_annotations.cleaned_v2.coco.json \
    --max-image-dim 4000 \
    --copy-paste \
    --class-weight-loss \
    --run-name "submission-banker-cleanedv2-maxdim4k-copypaste-classweight-fulltrain"
