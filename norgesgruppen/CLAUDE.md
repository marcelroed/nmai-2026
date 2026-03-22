Use Python only through `uv` with `uv run python`.

Make sure to specify which GPU to use for GPU runs using `CUDA_VISIBLE_DEVICES`. Check that the GPU is idle before launching.

Example training run:
```
CUDA_VISIBLE_DEVICES=1 uv run train --lr 3e-4 --epochs 200 --output-dir output-lr3e-4
```

## Eval
```bash
uv run python analyze_holdout_errors.py --checkpoint <ckpt.pth>         # full viz
uv run python analyze_holdout_errors.py --checkpoint <ckpt.pth> --no-viz # stats only
```

## Key info
- Model: RF-DETR 2XLarge, 880×880 patches, 440px overlap, NMS IoU 0.6
- 356 categories + 1 background (NUM_CLASSES+1=358 in model head)
- Scoring: 0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5
- Dataset: `data/train/images/` (248 images)
  - Original annotations: `data/train/annotations.json` (COCO `[x,y,w,h]`) — used for eval/splitting
  - Cleaned v1 annotations: `data/dataset/train/_annotations.coco.json` — name normalization (& → removed, GL.FRI → GLFRI)
  - Cleaned v2 annotations: `data/dataset/train/_annotations.cleaned_v2.coco.json` — deeper corrections (bbox fixes, category reassignments)
  - Use `--train-annotations` to train on cleaned, `--eval-annotations` for eval source
  - Use `--merge-labels` to merge MÜSLI variant categories into canonical ones (see `LABEL_MERGE_MAP` in `splitting.py`)
- Competition eval interval: 5 epochs (default)
- Config: `src/norgesgruppen/config.py` (MODEL_SIZE, resolution, patch config)
- See `overview.md` for detailed data analysis, error breakdowns, and improvement ideas
