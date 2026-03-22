#!/bin/bash
set -euo pipefail

# YOLOv26 XLarge — random crop training at 880px, matching RF-DETR patch setup
# Uses cleaned v2 annotations with label merging

CUDA_VISIBLE_DEVICES=0 \
uv run train-yolo \
    --yolo-model yolo26x.pt \
    --lr 0.01 \
    --epochs 200 \
    --batch-size 16 \
    --imgsz 880 \
    --output-dir output-yolo26x \
    --val-fraction 0.5 \
    --seed 42 \
    --competition-eval-interval 5 \
    --crop-mode random \
    --samples-per-epoch 5000 \
    --crop-scale-min 0.5 \
    --crop-scale-max 1.5 \
    --train-annotations data/dataset/train/_annotations.cleaned_v2.coco.json \
    --merge-labels \
    --run-name "yolo26x-880-random-crop"
