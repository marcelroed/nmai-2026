#!/bin/bash
# 001_verify_harness.sh — Quick verification that the training harness works end-to-end.
#
# Tests: stratified 50/50 split, wandb logging, full-image competition eval,
# wandb prediction visualizations.
#
# Uses 2XLarge model (880×880) with batch=8, no grad accum.
# 5 epochs, competition eval every 2 epochs.

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run train \
    --lr 1e-4 \
    --epochs 5 \
    --batch-size 8 \
    --grad-accum 1 \
    --output-dir output-verify-harness \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 2 \
    --model-size xxlarge
