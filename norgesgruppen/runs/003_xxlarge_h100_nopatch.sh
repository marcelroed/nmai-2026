#!/bin/bash
# 003_xxlarge_h100_nopatch.sh — Full training WITHOUT patching on H100 80GB.
#
# RF-DETR 2XLarge (880×880), stratified 50/50 split.
# No patching — full images resized to 880×880 by rfdetr's transform pipeline.
# This tests whether patching helps or hurts vs simple resize.
#
# batch=64, 200 epochs, competition eval every 10 epochs.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --grad-accum 1 \
    --output-dir output-xxl-h100-nopatch \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode resize \
    --run-name "no-patches-xxl-resize-only"
