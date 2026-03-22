"""Competition scoring: 70% detection mAP@0.5 + 30% classification mAP@0.5.

Usage:
    score = compute_score(ground_truths, predictions)

Where ground_truths and predictions are lists (one per image) of dicts:
    {"boxes": np.ndarray (N, 4) in xyxy, "labels": np.ndarray (N,)}
    predictions also has "scores": np.ndarray (N,)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreResult:
    detection_map: float
    classification_map: float
    combined: float

    def __str__(self) -> str:
        return (
            f"detection mAP@0.5: {self.detection_map:.4f}  "
            f"classification mAP@0.5: {self.classification_map:.4f}  "
            f"combined: {self.combined:.4f}"
        )


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of xyxy boxes. Returns (len(a), len(b))."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _ap_from_matches(
    scores: np.ndarray,
    matched: np.ndarray,
    n_gt: int,
) -> float:
    """Compute AP from sorted (by score) match indicators.

    Args:
        scores: confidence scores for each prediction (descending order).
        matched: boolean array, True if prediction matched a GT.
        n_gt: total number of ground truth boxes.
    """
    if n_gt == 0:
        return 0.0 if len(scores) > 0 else 1.0

    # Sort by descending score
    order = np.argsort(-scores)
    matched = matched[order]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)
    recall = tp / n_gt
    precision = tp / (tp + fp)

    # COCO-style: interpolate precision at each recall level
    # Prepend (0, 1) and append (1, 0)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _match_predictions(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    iou_threshold: float = 0.5,
    check_class: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Match predictions to ground truths, returning (scores, matched) arrays.

    Each GT can be matched at most once (greedy, by descending score).
    """
    if len(pred_boxes) == 0:
        return np.array([]), np.array([], dtype=bool)

    scores = pred_scores.copy()
    matched = np.zeros(len(pred_boxes), dtype=bool)

    if len(gt_boxes) == 0:
        return scores, matched

    iou_matrix = _compute_iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-scores)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    for pred_idx in order:
        ious = iou_matrix[pred_idx]
        # Mask already-matched GTs
        ious[gt_matched] = 0.0

        best_gt = np.argmax(ious)
        if ious[best_gt] >= iou_threshold:
            if not check_class or pred_labels[pred_idx] == gt_labels[best_gt]:
                matched[pred_idx] = True
                gt_matched[best_gt] = True

    return scores, matched


def compute_score(
    ground_truths: list[dict],
    predictions: list[dict],
    iou_threshold: float = 0.5,
) -> ScoreResult:
    """Compute the competition score.

    Args:
        ground_truths: list of {"boxes": (N,4) xyxy, "labels": (N,)}
        predictions: list of {"boxes": (N,4) xyxy, "labels": (N,), "scores": (N,)}
        iou_threshold: IoU threshold for matching.

    Returns:
        ScoreResult with detection_map, classification_map, and combined score.
    """
    assert len(ground_truths) == len(predictions)

    # Collect all per-image matches
    det_scores_all = []
    det_matched_all = []
    cls_scores_all = []
    cls_matched_all = []
    n_gt_total = 0

    # For per-class classification AP
    per_class_data: dict[int, dict] = {}  # label -> {"scores": [], "matched": [], "n_gt": int}

    for gt, pred in zip(ground_truths, predictions):
        gt_boxes = gt["boxes"] if len(gt["boxes"]) > 0 else np.zeros((0, 4))
        gt_labels = gt["labels"] if len(gt["labels"]) > 0 else np.zeros((0,), dtype=int)
        pred_boxes = pred["boxes"] if len(pred["boxes"]) > 0 else np.zeros((0, 4))
        pred_scores = pred["scores"] if len(pred["scores"]) > 0 else np.zeros((0,))
        pred_labels = pred["labels"] if len(pred["labels"]) > 0 else np.zeros((0,), dtype=int)

        n_gt_total += len(gt_boxes)

        # Detection: class-agnostic matching
        scores, matched = _match_predictions(
            gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
            iou_threshold=iou_threshold, check_class=False,
        )
        det_scores_all.append(scores)
        det_matched_all.append(matched)

        # Classification: class-aware matching, per class
        scores_cls, matched_cls = _match_predictions(
            gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
            iou_threshold=iou_threshold, check_class=True,
        )
        cls_scores_all.append(scores_cls)
        cls_matched_all.append(matched_cls)

        # Track per-class GT counts
        for label in gt_labels:
            if label not in per_class_data:
                per_class_data[label] = {"scores": [], "matched": [], "n_gt": 0}
            per_class_data[label]["n_gt"] += 1

        # Track per-class predictions
        for j in range(len(pred_labels)):
            label = int(pred_labels[j])
            if label not in per_class_data:
                per_class_data[label] = {"scores": [], "matched": [], "n_gt": 0}
            per_class_data[label]["scores"].append(scores_cls[j])
            per_class_data[label]["matched"].append(matched_cls[j])

    # Detection mAP: single-class AP (all boxes treated as one class)
    all_det_scores = np.concatenate(det_scores_all) if det_scores_all else np.array([])
    all_det_matched = np.concatenate(det_matched_all) if det_matched_all else np.array([], dtype=bool)
    detection_map = _ap_from_matches(all_det_scores, all_det_matched, n_gt_total)

    # Classification mAP: mean AP across classes that have at least one GT
    class_aps = []
    for label, data in per_class_data.items():
        if data["n_gt"] == 0:
            continue
        scores = np.array(data["scores"]) if data["scores"] else np.array([])
        matched = np.array(data["matched"], dtype=bool) if data["matched"] else np.array([], dtype=bool)
        class_aps.append(_ap_from_matches(scores, matched, data["n_gt"]))

    classification_map = float(np.mean(class_aps)) if class_aps else 0.0

    combined = 0.7 * detection_map + 0.3 * classification_map

    return ScoreResult(
        detection_map=detection_map,
        classification_map=classification_map,
        combined=combined,
    )
