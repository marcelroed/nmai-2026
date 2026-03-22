#!/bin/bash
# 002_xxlarge_h100_full.sh — Full training run on H100 80GB.
#
# RF-DETR 2XLarge (880×880), stratified 50/50 split.
# batch=64, no grad accum.
# 200 epochs, competition eval every 10 epochs.
# Cosine LR schedule with 3-epoch warmup.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --grad-accum 1 \
    --output-dir output-xxl-h100 \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --model-size xxlarge \
    --crop-mode fixed \
    --run-name "baseline-xxl-patched"
