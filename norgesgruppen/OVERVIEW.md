# NorgesGruppen Products — Competition Overview

## Task
Detect and classify grocery products on store shelf images.
Submit a `.zip` with `run.py` at root; it runs in a sandboxed Docker container.

```
python run.py --images /data/images/ --output /output/predictions.json
```

Output: JSON array of `{bbox: [x, y, w, h], category_id: int}`

---

## Scoring (hybrid)
| Component | Weight | Condition |
|-----------|--------|-----------|
| Detection mAP | 70% | IoU ≥ 0.5, category ignored |
| Classification mAP | 30% | IoU ≥ 0.5 AND correct category_id |

- Detection-only (all `category_id = 0`) caps out at 0.70
- **Implication: nailing detection first, then improving classification, is a natural two-stage strategy**

Submission limit: **2 per day** (resets midnight UTC). Deadline: **March 22, 15:00 CET**.

---

## Dataset
- **248 shelf images**, ~22,700 bounding box annotations, **356 product categories**
- Store sections: Egg, Frokost, Knekkebrod, Varmedrikke
- COCO format annotations: `[x, y, width, height]`
- Also available: **327 reference product images** (multi-angle, organized by barcode)

### Image resolution stats
- **96% of images are larger than 880×880** (the model's max input resolution)
- Average image is **10.4× the area** of 880×880 — massive information loss when downscaling
- Median resolution: 3024×3000, range: 481×399 to 5712×4284
- 114 unique resolutions — very heterogeneous
- Dominant: 4032×3024 (24%), 3024×4032 (11%), 4000×3000 (7%)
- These large images would need ~20 patches each at 880×880 with overlap
- ~92 boxes per image on average (range 14–235)

**Implication**: downscaling to 880×880 discards most of the detail. This likely explains the low recall (48.5%) — small products become undetectable. Patch-based inference is essential. — useful for classification

---

## Sandbox constraints
- GPU: NVIDIA L4 (24 GB VRAM), PyTorch 2.6.0+cu124
- Pre-installed: YOLOv8, ONNX Runtime, OpenCV, Albumentations
- **No runtime pip install**
- Blocked: `os`, `sys`, `subprocess`, `pickle`, `requests`, `multiprocessing`
- Max zip size: **420 MB**

---

## Data situation
- **248 images, ~22,700 bounding boxes, 356 classes** — very small dataset
- Aggressive augmentation is essential to avoid overfitting
- **No held-out validation split** — we train on all data and use 3LC metrics on the full training set for model insight instead. Losing even 10-20% to a val split is too costly given the size.

---

## Core workflow: data-driven iteration with 3LC

3LC is our central tool for understanding and improving the dataset/model loop.

```
[Create 3LC table once from raw COCO data]
        |
        v
  ┌─────────────────────────────────────────┐
  │  1. Fetch latest 3LC table revision     │
  │  2. Export to COCO for training         │
  │  3. Train iteration model (RF-DETR L)   │
  │  4. Run full inference → collect 3LC    │
  │     metrics (BoundingBoxMetricsCollector│
  │     + embeddings)                       │
  │  5. Analyze in 3LC dashboard:           │
  │     - Find mislabeled / hard examples   │
  │     - Adjust sample weights             │
  │     - Fix / remove bad annotations      │
  │  6. New table revision created          │
  └────────────────┬────────────────────────┘
                   │
             repeat loop
                   │
                   v
     [Final submission model: RF-DETR 2XLarge
      trained on best table revision]
```

**Iteration model**: RF-DETR Large — fast enough to iterate, same architecture family as submission model.
**Submission model**: RF-DETR 2XLarge — trained on the final, curated table revision.

---

## Ideas for improvement

### Data quality (highest leverage given small dataset)
- Use 3LC metrics to identify mislabeled boxes (high loss, wrong predicted class)
- Use 3LC embeddings to find duplicate/near-duplicate images or annotation errors
- Upweight hard/rare examples via 3LC sample weights
- Carefully review low-confidence predictions in the dashboard

### Co-occurrence / shelf context
Grocery shelves are organized: similar products are stocked together. We can exploit this:
- **Build co-occurrence lists** from training annotations: for each image, record which category_ids appear together. Cluster these into "shelf groups" (e.g. eggs, cereals, crispbread, hot drinks — matching the 4 store sections).
- **Label cleaning**: if an image has 200 boxes from one shelf group and 2 from another, those 2 are likely mislabeled — flag for review and potential removal.
- **Inference-time filtering**: given the predicted distribution of categories in a test image, identify the dominant shelf group and suppress low-confidence predictions from other groups.
- First cut of the lists comes for free by mining the training annotations; can be refined manually.

### Patch-based training & inference
Model max input resolution is **880×880**, but shelf images are larger. Strategy:
- **Training**: extract 880×880 patches from images (with overlap to avoid cutting objects at borders); each patch is a training sample — this also multiplies the effective dataset size.
- **Inference**: tile the full image into overlapping 880×880 patches, run detection on each, then stitch predictions back to full-image coordinates using NMS to resolve duplicates across patch boundaries.
- Overlap between patches should be at least ~20% to ensure objects near borders appear fully in at least one patch.
- This means data augmentation like random crops is also naturally handled by the patch extraction strategy.

### Augmentation
- Mosaic, copy-paste, color jitter — critical with only 248 images
- Random scale/flip/rotate within patches
- Mixup for bounding boxes

### Detection (70%)
- More epochs with full dataset (no val split wasted)
- TTA at inference: flips, multi-scale
- Ensemble multiple checkpoints

### Classification (30%)
- Leverage **reference product images** (327 products, multi-angle) — these are free supervision
- Two-stage: detect first (category_id=0), then classify crops against reference images using embeddings
- CLIP or a fine-tuned embedding model to match crops to barcoded reference images

### Submission
- Fix ONNX export pipeline (`prepare_model.py` export is currently commented out)
- `run.py` must avoid blocked imports (`os`, `sys`, `subprocess`, `pickle`, `requests`, `multiprocessing`)
- Profile inference time — all 248 test images on L4 GPU

---

## How we work
- Iterate fast on RF-DETR Large; scale up to 2XLarge for submissions
- Each 3LC loop = one potential improvement to the dataset
- 2 submissions/day — use them deliberately after meaningful changes
- Log all submission scores below

### Code design principle: composability
Keep ideas as small, independent, pluggable functions/transforms. Every step in the pipeline — patch extraction, augmentation, co-occurrence filtering, inference-time NMS stitching, post-processing — should be a self-contained component that can be enabled/disabled independently. This lets us quickly A/B test ideas without large refactors and makes it easy to isolate what actually moves the score.

---

## Results log
| Date | Model | Notes | Det mAP | Cls mAP | Combined |
|------|-------|-------|---------|---------|----------|
| Mar 20 | RFDETRLarge | 10ep, holdout 2-fold avg, full-image resize to 880 | 0.535 | 0.232 | 0.444 |

### Baseline analysis (fold 0 holdout, 15 images)
- Precision: 93.1%, **Recall: 48.5%** — FN is the dominant problem
- 741 TP, 55 FP, 787 FN out of 1528 GT boxes
- Classification accuracy among TPs: 65.3%
- FPs are low confidence (mean 0.38), TPs are moderate (mean 0.47)
- **Main bottleneck: recall.** Half the products are being missed, almost certainly due to downscaling large images to 880×880.
