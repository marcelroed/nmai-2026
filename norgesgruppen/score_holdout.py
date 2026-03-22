#!/usr/bin/env python3
"""Run the competition scoring function on holdout images and analyze score breakdown.

Usage:
    uv run python score_holdout.py --checkpoint checkpoint_1e-3_train_split.pth
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from norgesgruppen.config import MODEL_CLS, NUM_CLASSES
from norgesgruppen.patching import predict_with_patches
from norgesgruppen.postprocess import DEFAULT_TRANSFORMS, apply_transforms
from norgesgruppen.scoring import (
    ScoreResult,
    _ap_from_matches,
    _compute_iou_matrix,
    _match_predictions,
    compute_score,
)

HOLDOUT_IMAGES = [
    "img_00005.jpg", "img_00012.jpg", "img_00026.jpeg", "img_00036.jpg",
    "img_00062.jpg", "img_00075.jpg", "img_00088.jpg", "img_00105.jpg",
    "img_00108.jpg", "img_00122.jpg", "img_00152.jpeg", "img_00163.jpg",
    "img_00168.jpg", "img_00178.jpg", "img_00184.jpeg", "img_00193.jpg",
    "img_00243.jpg", "img_00261.jpg", "img_00271.jpg", "img_00296.jpg",
    "img_00302.jpg", "img_00304.jpg", "img_00317.jpg", "img_00320.jpg",
    "img_00373.jpg",
]
IMAGES_DIR = Path("data/train/images")
ANNOTATIONS_PATH = Path("data/train/annotations.json")


def load_gt():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    img_by_fname = {img["file_name"]: img for img in data["images"]}
    anns_by_imgid = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_imgid[ann["image_id"]].append(ann)

    gt_by_fname = {}
    for fname, img_info in img_by_fname.items():
        anns = anns_by_imgid[img_info["id"]]
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        gt_by_fname[fname] = {
            "boxes": np.array(boxes).reshape(-1, 4),
            "labels": np.array(labels, dtype=int),
        }
    return cat_names, gt_by_fname


def load_model(checkpoint_path):
    import argparse as _argparse
    model = MODEL_CLS()
    model.model.reinitialize_detection_head(num_classes=NUM_CLASSES + 1)
    torch.serialization.add_safe_globals([_argparse.Namespace])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.model.model.load_state_dict(ckpt[state_key])
    print(f"Loaded {state_key} weights (epoch {ckpt.get('epoch', '?')})")
    model.model.model.eval().cuda()
    return model


def predict_single(model, image, threshold=0.2):
    dets = predict_with_patches(image, lambda img: model.predict(img, threshold=threshold))
    return apply_transforms(dets, DEFAULT_TRANSFORMS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--predict-threshold", type=float, default=0.2)
    args = parser.parse_args()

    cat_names, gt_by_fname = load_gt()
    model = load_model(args.checkpoint)

    ground_truths = []
    predictions = []

    for img_name in HOLDOUT_IMAGES:
        image_path = IMAGES_DIR / img_name
        if not image_path.exists() or img_name not in gt_by_fname:
            continue

        print(f"  {img_name}...", end=" ", flush=True)
        gt = gt_by_fname[img_name]
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad(), torch.amp.autocast("cuda"):
            dets = predict_single(model, image, threshold=args.predict_threshold)

        if dets.is_empty():
            pred = {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,))}
        else:
            pred = {"boxes": dets.xyxy, "labels": dets.class_id, "scores": dets.confidence}

        ground_truths.append(gt)
        predictions.append(pred)
        print(f"GT={len(gt['boxes'])} Pred={len(pred['boxes'])}")

    # =====================================================================
    # 1. Overall competition score
    # =====================================================================
    result = compute_score(ground_truths, predictions)
    print(f"\n{'='*80}")
    print(f"COMPETITION SCORE")
    print(f"{'='*80}")
    print(f"  Detection mAP@0.5:       {result.detection_map:.4f}  (weight: 0.7 → contributes {0.7*result.detection_map:.4f})")
    print(f"  Classification mAP@0.5:  {result.classification_map:.4f}  (weight: 0.3 → contributes {0.3*result.classification_map:.4f})")
    print(f"  Combined score:          {result.combined:.4f}")
    print(f"  Perfect score:           1.0000")
    print(f"  Gap to perfect:          {1.0 - result.combined:.4f}")
    det_gap = 0.7 * (1.0 - result.detection_map)
    cls_gap = 0.3 * (1.0 - result.classification_map)
    print(f"    from detection:        {det_gap:.4f}  ({100*det_gap/(det_gap+cls_gap):.1f}% of gap)")
    print(f"    from classification:   {cls_gap:.4f}  ({100*cls_gap/(det_gap+cls_gap):.1f}% of gap)")

    # =====================================================================
    # 2. Decompose detection mAP loss: FP vs FN
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"DETECTION mAP DECOMPOSITION")
    print(f"{'='*80}")

    n_gt_total = sum(len(gt["boxes"]) for gt in ground_truths)
    all_det_scores = []
    all_det_matched = []
    for gt, pred in zip(ground_truths, predictions):
        scores, matched = _match_predictions(
            gt["boxes"], gt["labels"], pred["boxes"], pred["scores"], pred["labels"],
            iou_threshold=0.5, check_class=False,
        )
        all_det_scores.append(scores)
        all_det_matched.append(matched)

    det_scores = np.concatenate(all_det_scores) if all_det_scores else np.array([])
    det_matched = np.concatenate(all_det_matched) if all_det_matched else np.array([], dtype=bool)

    n_det_tp = int(det_matched.sum())
    n_det_fp = int((~det_matched).sum())
    n_det_fn = n_gt_total - n_det_tp

    print(f"  Total GT boxes:    {n_gt_total}")
    print(f"  Total predictions: {len(det_scores)}")
    print(f"  True positives:    {n_det_tp}  (matched a GT at IoU≥0.5)")
    print(f"  False positives:   {n_det_fp}  (no GT match)")
    print(f"  False negatives:   {n_det_fn}  (GT not matched by any pred)")
    print(f"  Precision:         {n_det_tp/max(len(det_scores),1):.4f}")
    print(f"  Recall:            {n_det_tp/max(n_gt_total,1):.4f}")

    # What if we had perfect precision (no FPs)?
    perfect_prec_matched = det_matched[det_matched]  # only TPs
    perfect_prec_scores = det_scores[det_matched]
    det_map_no_fp = _ap_from_matches(perfect_prec_scores, perfect_prec_matched, n_gt_total)

    # What if we had perfect recall (no FNs) but same FP rate?
    # Can't easily simulate, but we can report the ceiling
    print(f"\n  Detection mAP if all FPs removed:  {det_map_no_fp:.4f}  (gain: +{det_map_no_fp - result.detection_map:.4f})")
    print(f"  → FPs cost us {result.detection_map - det_map_no_fp:.4f} det mAP ({0.7*(det_map_no_fp - result.detection_map):.4f} combined)")
    print(f"  → FNs cost us {1.0 - det_map_no_fp:.4f} det mAP ({0.7*(1.0 - det_map_no_fp):.4f} combined)")
    print(f"     (the remaining gap if we had zero FPs is entirely from missed GT)")

    # =====================================================================
    # 3. Decompose classification mAP: per-class breakdown
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION mAP DECOMPOSITION")
    print(f"{'='*80}")

    per_class_data = defaultdict(lambda: {"scores": [], "matched": [], "n_gt": 0})
    for gt, pred in zip(ground_truths, predictions):
        gt_boxes = gt["boxes"] if len(gt["boxes"]) > 0 else np.zeros((0, 4))
        gt_labels = gt["labels"] if len(gt["labels"]) > 0 else np.zeros((0,), dtype=int)
        pred_boxes = pred["boxes"] if len(pred["boxes"]) > 0 else np.zeros((0, 4))
        pred_scores = pred["scores"] if len(pred["scores"]) > 0 else np.zeros((0,))
        pred_labels = pred["labels"] if len(pred["labels"]) > 0 else np.zeros((0,), dtype=int)

        for lbl in gt_labels:
            per_class_data[int(lbl)]["n_gt"] += 1

        scores_cls, matched_cls = _match_predictions(
            gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
            iou_threshold=0.5, check_class=True,
        )
        for j in range(len(pred_labels)):
            lbl = int(pred_labels[j])
            per_class_data[lbl]["scores"].append(scores_cls[j])
            per_class_data[lbl]["matched"].append(matched_cls[j])

    # Compute per-class AP
    class_aps = {}
    for lbl, d in per_class_data.items():
        if d["n_gt"] == 0:
            continue
        scores = np.array(d["scores"]) if d["scores"] else np.array([])
        matched = np.array(d["matched"], dtype=bool) if d["matched"] else np.array([], dtype=bool)
        class_aps[lbl] = _ap_from_matches(scores, matched, d["n_gt"])

    n_classes_evaluated = len(class_aps)
    mean_cls_ap = np.mean(list(class_aps.values()))
    print(f"  Classes evaluated (≥1 GT in holdout): {n_classes_evaluated}")
    print(f"  Mean per-class AP (cls mAP@0.5):      {mean_cls_ap:.4f}")

    # AP distribution
    ap_values = list(class_aps.values())
    print(f"\n  AP distribution:")
    for threshold in [1.0, 0.9, 0.8, 0.5, 0.1, 0.0]:
        count = sum(1 for ap in ap_values if ap >= threshold)
        print(f"    AP ≥ {threshold:.1f}: {count:>3} classes ({100*count/n_classes_evaluated:.1f}%)")

    zero_ap = [(lbl, ap) for lbl, ap in class_aps.items() if ap == 0.0]
    perfect_ap = [(lbl, ap) for lbl, ap in class_aps.items() if ap == 1.0]
    print(f"\n  Classes with AP = 0.0: {len(zero_ap)}")
    print(f"  Classes with AP = 1.0: {len(perfect_ap)}")

    # Impact analysis: what if we fixed the worst classes?
    sorted_by_ap = sorted(class_aps.items(), key=lambda x: x[1])

    print(f"\n  --- If we raised the WORST N classes to AP=1.0: ---")
    for n_fix in [5, 10, 20, 50]:
        fixed = list(class_aps.values())
        for i, (lbl, ap) in enumerate(sorted_by_ap):
            if i < n_fix:
                fixed[list(class_aps.keys()).index(lbl)] = 1.0
        new_mean = np.mean(fixed)
        new_combined = 0.7 * result.detection_map + 0.3 * new_mean
        print(f"    Fix worst {n_fix:>2}: cls_mAP {new_mean:.4f} (+{new_mean - mean_cls_ap:.4f}), combined {new_combined:.4f} (+{new_combined - result.combined:.4f})")

    # Bottom 30 classes by AP
    print(f"\n{'='*80}")
    print(f"BOTTOM 30 CLASSES BY AP (biggest score drags)")
    print(f"{'='*80}")
    print(f"{'Cat':>4} {'AP':>6} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4}  Name")
    print("-" * 80)
    for lbl, ap in sorted_by_ap[:30]:
        d = per_class_data[lbl]
        n_gt = d["n_gt"]
        matched = np.array(d["matched"], dtype=bool) if d["matched"] else np.array([], dtype=bool)
        n_tp = int(matched.sum())
        n_fp = int((~matched).sum()) if len(matched) > 0 else 0
        n_fn = n_gt - n_tp
        name = cat_names.get(lbl, f"cat_{lbl}")[:45]
        print(f"{lbl:>4} {ap:>6.3f} {n_gt:>4} {n_tp:>4} {n_fp:>4} {n_fn:>4}  {name}")

    # =====================================================================
    # 4. Score sensitivity: what threshold is best?
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"SCORE VS CONFIDENCE THRESHOLD")
    print(f"{'='*80}")
    for thr in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        filtered_preds = []
        for pred in predictions:
            if len(pred["scores"]) == 0:
                filtered_preds.append(pred)
                continue
            mask = pred["scores"] >= thr
            filtered_preds.append({
                "boxes": pred["boxes"][mask],
                "labels": pred["labels"][mask],
                "scores": pred["scores"][mask],
            })
        r = compute_score(ground_truths, filtered_preds)
        n_preds = sum(len(p["scores"]) for p in filtered_preds)
        print(f"  conf≥{thr:.2f}: det={r.detection_map:.4f} cls={r.classification_map:.4f} combined={r.combined:.4f}  ({n_preds} preds)")


if __name__ == "__main__":
    main()
