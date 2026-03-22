#!/usr/bin/env python3
"""Analyze and visualize prediction errors on holdout images.

Usage (CLI):
    uv run python analyze_holdout_errors.py --checkpoint path/to/ckpt.pth
    uv run python analyze_holdout_errors.py --checkpoint ckpt.pth --conf 0.5 --save-dir my_viz
    uv run python analyze_holdout_errors.py --checkpoint ckpt.pth --no-viz   # stats only

Usage (programmatic):
    from analyze_holdout_errors import run_analysis
    results = run_analysis("checkpoint_1e-3_train_split.pth")
    print(results["summary"])

Color code:
  GREEN  = correct (IoU >= 0.5, correct class)
  YELLOW = localized but misclassified (IoU >= 0.5, wrong class)
  RED    = false positive (no GT match)
  BLUE   = missed ground truth (false negative)
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from norgesgruppen.config import MODEL_CLS, NUM_CLASSES
from norgesgruppen.patching import predict_with_patches
from norgesgruppen.postprocess import DEFAULT_TRANSFORMS, apply_transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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

# Colors (BGR for OpenCV)
GREEN = (0, 200, 0)
YELLOW = (0, 220, 255)
RED = (0, 0, 220)
BLUE = (255, 160, 0)


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResults:
    """Structured results from a holdout error analysis run."""
    checkpoint: str
    conf_threshold: float
    iou_threshold: float
    predict_threshold: float

    total_gt: int = 0
    total_pred: int = 0
    total_correct: int = 0
    total_misclassified: int = 0
    total_fp: int = 0
    total_missed: int = 0

    per_image_stats: list = field(default_factory=list)
    confusion_pairs: Counter = field(default_factory=Counter)
    gt_cat_correct: Counter = field(default_factory=Counter)
    gt_cat_missed: Counter = field(default_factory=Counter)
    gt_cat_misclassified: Counter = field(default_factory=Counter)
    gt_cat_total: Counter = field(default_factory=Counter)
    fp_by_cat: Counter = field(default_factory=Counter)

    @property
    def precision(self):
        return self.total_correct / max(self.total_pred, 1)

    @property
    def recall(self):
        return self.total_correct / max(self.total_gt, 1)

    @property
    def summary(self) -> str:
        lines = [
            f"Checkpoint: {self.checkpoint}",
            f"Thresholds: conf>={self.conf_threshold}, IoU>={self.iou_threshold}, predict>={self.predict_threshold}",
            f"GT boxes: {self.total_gt}  |  Predictions: {self.total_pred}",
            f"Correct: {self.total_correct} ({100*self.total_correct/max(self.total_pred,1):.1f}% of preds, {100*self.total_correct/max(self.total_gt,1):.1f}% of GT)",
            f"Misclassified: {self.total_misclassified} ({100*self.total_misclassified/max(self.total_pred,1):.1f}%)",
            f"False Positives: {self.total_fp} ({100*self.total_fp/max(self.total_pred,1):.1f}%)",
            f"Missed (FN): {self.total_missed} ({100*self.total_missed/max(self.total_gt,1):.1f}%)",
            f"Precision: {self.precision:.3f}  |  Recall: {self.recall:.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core logic (stateless functions)
# ---------------------------------------------------------------------------

def load_annotations(
    annotations_path: Path = ANNOTATIONS_PATH,
) -> tuple[dict[int, str], dict[str, dict]]:
    """Load COCO annotations, return (category_names, gt_by_filename)."""
    with open(annotations_path) as f:
        data = json.load(f)

    cat_names = {cat["id"]: cat["name"] for cat in data["categories"]}
    img_by_fname = {img["file_name"]: img for img in data["images"]}
    anns_by_imgid: dict[int, list] = defaultdict(list)
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


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_and_classify(gt, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """Match predictions to GT. Returns (status, gt_match, gt_label, iou, missed_idx)."""
    gt_boxes, gt_labels = gt["boxes"], gt["labels"]
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)

    pred_status = ["fp"] * n_pred
    pred_gt_match = [None] * n_pred
    pred_gt_label = [None] * n_pred
    pred_iou = [0.0] * n_pred

    if n_gt == 0 or n_pred == 0:
        return pred_status, pred_gt_match, pred_gt_label, pred_iou, list(range(n_gt))

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    gt_matched = np.zeros(n_gt, dtype=bool)

    for pred_idx in np.argsort(-pred_scores):
        ious = iou_matrix[pred_idx].copy()
        ious[gt_matched] = 0.0
        best_gt = np.argmax(ious)
        best_iou = ious[best_gt]
        pred_iou[pred_idx] = best_iou

        if best_iou >= iou_thresh:
            gt_matched[best_gt] = True
            pred_gt_match[pred_idx] = int(best_gt)
            pred_gt_label[pred_idx] = int(gt_labels[best_gt])
            if pred_labels[pred_idx] == gt_labels[best_gt]:
                pred_status[pred_idx] = "correct"
            else:
                pred_status[pred_idx] = "misclassified"

    missed_gt_idx = [i for i in range(n_gt) if not gt_matched[i]]
    return pred_status, pred_gt_match, pred_gt_label, pred_iou, missed_gt_idx


def load_model(checkpoint_path: str):
    """Load RF-DETR model from a .pth checkpoint."""
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


def predict_single(model, image: Image.Image, threshold: float = 0.2):
    """Run patch-based inference + post-processing on a single PIL image."""
    dets = predict_with_patches(image, lambda img: model.predict(img, threshold=threshold))
    return apply_transforms(dets, DEFAULT_TRANSFORMS)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_label(img, text, x, y, color, above=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.38, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if above:
        cv2.rectangle(img, (x, y - th - 4), (x + tw, y), color, -1)
        cv2.putText(img, text, (x, y - 2), font, scale, (0, 0, 0), thick)
    else:
        cv2.rectangle(img, (x, y), (x + tw, y + th + 4), color, -1)
        cv2.putText(img, text, (x, y + th + 2), font, scale, (0, 0, 0), thick)


def draw_error_viz(image_path, pred_boxes, pred_labels, pred_scores, pred_status,
                   pred_gt_label, pred_iou, missed_gt_boxes, missed_gt_labels,
                   cat_names, conf_threshold):
    """Draw color-coded error boxes on the image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Missed GT (blue)
    for i in range(len(missed_gt_boxes)):
        x1, y1, x2, y2 = missed_gt_boxes[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, 3)
        text = f"MISSED: {cat_names.get(int(missed_gt_labels[i]), '?')[:25]}"
        _draw_label(img, text, x1, y2 + 2, BLUE, above=False)

    # Predictions
    for i in range(len(pred_boxes)):
        if pred_scores[i] < conf_threshold:
            continue
        x1, y1, x2, y2 = pred_boxes[i].astype(int)
        status = pred_status[i]
        cat_id = pred_labels[i]
        conf = pred_scores[i]
        iou = pred_iou[i]

        if status == "correct":
            color, thickness = GREEN, 2
            text = f"{cat_names.get(cat_id, str(cat_id))[:22]} {conf:.2f}"
        elif status == "misclassified":
            color, thickness = YELLOW, 3
            pred_name = cat_names.get(cat_id, str(cat_id))[:18]
            gt_name = cat_names.get(pred_gt_label[i], str(pred_gt_label[i]))[:18]
            text = f"PRED:{pred_name} GT:{gt_name} iou={iou:.2f}"
        else:
            color, thickness = RED, 2
            text = f"FP: {cat_names.get(cat_id, str(cat_id))[:20]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        _draw_label(img, text, x1, y1, color, above=True)

    return img


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: AnalysisResults, cat_names: dict[int, str]):
    """Print the full error analysis report to stdout."""
    r = results

    print("\n" + "=" * 80)
    print("AGGREGATE ERROR ANALYSIS")
    print("=" * 80)
    print(r.summary)

    # Per-image table
    print(f"\n{'Image':<22} {'GT':>4} {'Pred':>5} {'Corr':>5} {'Miscl':>6} {'FP':>4} {'Miss':>5} {'Recall':>7} {'Prec':>7}")
    print("-" * 80)
    for s in r.per_image_stats:
        rec = s["correct"] / max(s["gt"], 1)
        pre = s["correct"] / max(s["pred"], 1)
        print(f"{s['image']:<22} {s['gt']:>4} {s['pred']:>5} {s['correct']:>5} {s['misclassified']:>6} {s['fp']:>4} {s['missed']:>5} {rec:>7.3f} {pre:>7.3f}")

    # Confusion pairs
    print(f"\n{'=' * 80}\nTOP 25 CONFUSION PAIRS (GT → Predicted)\n{'=' * 80}")
    for (gt_cat, pred_cat), count in r.confusion_pairs.most_common(25):
        gt_n = cat_names.get(gt_cat, str(gt_cat))[:35]
        pr_n = cat_names.get(pred_cat, str(pred_cat))[:35]
        print(f"  {count:>3}x  {gt_n:<35} → {pr_n:<35}")

    # Most-missed
    print(f"\n{'=' * 80}\nTOP 20 MOST-MISSED CATEGORIES\n{'=' * 80}")
    for cat, n_miss in r.gt_cat_missed.most_common(20):
        n_total = r.gt_cat_total[cat]
        n_corr = r.gt_cat_correct[cat]
        rec = n_corr / max(n_total, 1)
        print(f"  {cat_names.get(cat, str(cat))[:40]:<40}  missed={n_miss:>3}/{n_total:<3}  recall={rec:.2f}")

    # Most-misclassified
    print(f"\n{'=' * 80}\nTOP 20 MOST-MISCLASSIFIED GT CATEGORIES\n{'=' * 80}")
    for cat, n_mis in r.gt_cat_misclassified.most_common(20):
        n_total = r.gt_cat_total[cat]
        print(f"  {cat_names.get(cat, str(cat))[:40]:<40}  misclassified={n_mis:>3}/{n_total:<3}")

    # Most FPs
    print(f"\n{'=' * 80}\nTOP 20 CATEGORIES WITH MOST FALSE POSITIVES\n{'=' * 80}")
    for cat, n_fp in r.fp_by_cat.most_common(20):
        print(f"  {cat_names.get(cat, str(cat))[:45]:<45}  FP={n_fp:>3}")

    # Worst recall
    print(f"\n{'=' * 80}\nTOP 20 WORST RECALL (min 2 GT instances)\n{'=' * 80}")
    cat_recall = []
    for cat, n_total in r.gt_cat_total.items():
        if n_total >= 2:
            n_corr = r.gt_cat_correct[cat]
            cat_recall.append((cat, n_corr / n_total, n_corr, n_total))
    cat_recall.sort(key=lambda x: (x[1], -x[3]))
    for cat, rec, n_corr, n_total in cat_recall[:20]:
        print(f"  {cat_names.get(cat, str(cat))[:40]:<40}  recall={rec:.2f}  ({n_corr}/{n_total})")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_analysis(
    checkpoint: str,
    save_dir: str | Path | None = None,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    predict_threshold: float = 0.2,
    save_viz: bool = True,
    print_results: bool = True,
) -> AnalysisResults:
    """Run full holdout error analysis. Returns structured AnalysisResults.

    Args:
        checkpoint: Path to .pth checkpoint file.
        save_dir: Directory for visualizations. Defaults to viz_errors_<checkpoint_stem>.
        conf_threshold: Min confidence to count a prediction.
        iou_threshold: IoU threshold for matching predictions to GT.
        predict_threshold: Raw model prediction threshold (before conf filter).
        save_viz: Whether to save error visualization images.
        print_results: Whether to print the report to stdout.

    Returns:
        AnalysisResults dataclass with all metrics.
    """
    # Resolve save dir
    if save_dir is None:
        ckpt_stem = Path(checkpoint).stem
        save_dir = Path(f"viz_errors_{ckpt_stem}")
    save_dir = Path(save_dir)

    cat_names, gt_by_fname = load_annotations()
    print(f"Loaded {len(cat_names)} categories, {len(gt_by_fname)} image GTs")

    print(f"Loading model from {checkpoint}...")
    model = load_model(checkpoint)

    if save_viz:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = AnalysisResults(
        checkpoint=checkpoint,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        predict_threshold=predict_threshold,
    )

    for img_name in HOLDOUT_IMAGES:
        image_path = IMAGES_DIR / img_name
        if not image_path.exists():
            print(f"  SKIP: {img_name}")
            continue
        gt = gt_by_fname.get(img_name)
        if gt is None:
            print(f"  NO GT: {img_name}")
            continue

        print(f"  Processing: {img_name}...", end=" ", flush=True)
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            dets = predict_single(model, image, threshold=predict_threshold)

        if dets.is_empty():
            pred_boxes = np.zeros((0, 4))
            pred_labels = np.zeros((0,), dtype=int)
            pred_scores = np.zeros((0,))
        else:
            pred_boxes = dets.xyxy
            pred_labels = dets.class_id
            pred_scores = dets.confidence

        # Filter to conf threshold
        mask = pred_scores >= conf_threshold
        pb, pl, ps = pred_boxes[mask], pred_labels[mask], pred_scores[mask]

        status, gt_match, gt_label, ious, missed_idx = match_and_classify(
            gt, pb, pl, ps, iou_threshold
        )

        n_correct = status.count("correct")
        n_misclass = status.count("misclassified")
        n_fp = status.count("fp")
        n_missed = len(missed_idx)
        n_gt = len(gt["boxes"])
        n_pred = len(pb)

        # Accumulate
        results.total_correct += n_correct
        results.total_misclassified += n_misclass
        results.total_fp += n_fp
        results.total_missed += n_missed
        results.total_gt += n_gt
        results.total_pred += n_pred

        for j, s in enumerate(status):
            cat = int(pl[j])
            if s == "correct":
                results.gt_cat_correct[gt_label[j]] += 1
            elif s == "misclassified":
                results.gt_cat_misclassified[gt_label[j]] += 1
                results.confusion_pairs[(gt_label[j], cat)] += 1
            else:
                results.fp_by_cat[cat] += 1

        for idx in missed_idx:
            results.gt_cat_missed[int(gt["labels"][idx])] += 1
        for lbl in gt["labels"]:
            results.gt_cat_total[int(lbl)] += 1

        results.per_image_stats.append({
            "image": img_name, "gt": n_gt, "pred": n_pred,
            "correct": n_correct, "misclassified": n_misclass,
            "fp": n_fp, "missed": n_missed,
        })

        print(f"GT={n_gt} Pred={n_pred} | correct={n_correct} misclass={n_misclass} FP={n_fp} missed={n_missed}")

        # Visualization
        if save_viz:
            missed_gt_boxes = gt["boxes"][missed_idx] if missed_idx else np.zeros((0, 4))
            missed_gt_labels = gt["labels"][missed_idx] if missed_idx else np.zeros((0,), dtype=int)
            viz = draw_error_viz(
                image_path, pb, pl, ps, status, gt_label, ious,
                missed_gt_boxes, missed_gt_labels, cat_names, conf_threshold,
            )
            if viz is not None:
                out_path = save_dir / f"{Path(img_name).stem}_errors.jpg"
                cv2.imwrite(str(out_path), viz)

    if print_results:
        print_report(results, cat_names)
        if save_viz:
            print(f"\nVisualizations saved to {save_dir}/")
            print("Legend: GREEN=correct, YELLOW=misclassified, RED=false positive, BLUE=missed GT")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Holdout error analysis for RF-DETR checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--save-dir", type=str, default=None, help="Output dir (default: viz_errors_<ckpt_stem>)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    parser.add_argument("--predict-threshold", type=float, default=0.2, help="Model predict threshold (default: 0.2)")
    parser.add_argument("--no-viz", action="store_true", help="Skip saving visualizations (stats only)")
    args = parser.parse_args()

    run_analysis(
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        predict_threshold=args.predict_threshold,
        save_viz=not args.no_viz,
    )


if __name__ == "__main__":
    main()
