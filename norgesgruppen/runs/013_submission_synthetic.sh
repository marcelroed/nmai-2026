#!/bin/bash
# 013_submission_synthetic.sh — Full-train submission with synthetic data + copy-paste.
#
# Best practices from runs 009/010 plus synthetic data:
#   - Cleaned v2 labels for training
#   - maxdim 4000px (downscale large images)
#   - Random crops with scale 0.5-1.5x
#   - Copy-paste augmentation (instance bank, confused-pair biased)
#   - 50/50 interleaving with synthetic shelf images (200 images, 16K anns,
#     321 categories uniform, FP hard negatives interleaved)
#   - Full train (no val split) — all 248 real images
#   - 200 epochs, batch=64, lr=1e-3
#
# After training, export with:
#   uv run prepare-submission --checkpoint output-submission-synthetic/checkpoint_best_total.pth
#
# IMPORTANT: run.py MAX_IMAGE_DIM must be set to 4000 to match training.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --output-dir output-submission-synthetic \
    --full-train \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode random \
    --train-annotations data/dataset/train/_annotations.cleaned_v2.coco.json \
    --max-image-dim 4000 \
    --copy-paste \
    --synthetic-data data/synthetic_shelves \
    --synthetic-weight 0.5 \
    --run-name "submission-synthetic50-cleanedv2-copypaste-fulltrain"
