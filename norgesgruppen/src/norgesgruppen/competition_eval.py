"""Full-image competition evaluation with patched inference.

Runs the raw LWDETR model on full-resolution validation images using patch-based
inference (matching actual submission behavior), then computes the competition
score: 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from norgesgruppen.patching import predict_with_patches
from norgesgruppen.scoring import ScoreResult, _compute_iou_matrix, compute_score


def build_predict_fn(
    model: torch.nn.Module,
    postprocess,
    resolution: int = 880,
    threshold: float = 0.2,
    device: torch.device = torch.device("cuda"),
) -> Callable[[Image.Image], sv.Detections]:
    """Create a predict function compatible with predict_with_patches().

    Args:
        model: Raw LWDETR model (not the RFDETR wrapper).
        postprocess: PostProcess dict from build_criterion_and_postprocessors.
            Should be postprocess["bbox"].
        resolution: Model input resolution.
        threshold: Confidence threshold.
        device: Torch device.

    Returns:
        Callable that takes a PIL Image and returns sv.Detections.
    """
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    def predict_fn(image: Image.Image) -> sv.Detections:
        w, h = image.size

        # PIL → tensor → normalize → resize → batch
        img_tensor = TF.to_tensor(image).to(device)  # [3, H, W] float32 in [0, 1]
        img_tensor = (img_tensor - mean) / std
        img_tensor = TF.resize(img_tensor, [resolution, resolution], antialias=True)
        batch = img_tensor.unsqueeze(0)  # [1, 3, H, W]

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            outputs = model(batch)

        # PostProcess expects orig_sizes as [B, 2] tensor (h, w)
        orig_sizes = torch.tensor([[h, w]], device=device)
        results = postprocess(outputs, orig_sizes)

        result = results[0]
        boxes = result["boxes"].cpu().numpy()  # [N, 4] xyxy
        scores = result["scores"].cpu().numpy()  # [N]
        labels = result["labels"].cpu().numpy()  # [N]

        # Filter by threshold
        keep = scores >= threshold
        if not keep.any():
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=boxes[keep],
            confidence=scores[keep],
            class_id=labels[keep],
        )

    return predict_fn


def _coco_to_gt(val_coco: dict) -> dict[int, dict]:
    """Convert COCO annotations to per-image GT dicts for scoring.

    Returns:
        {image_id: {"boxes": ndarray (N, 4) xyxy, "labels": ndarray (N,)}}
    """
    from collections import defaultdict

    anns_by_image = defaultdict(list)
    for ann in val_coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    gt_dict = {}
    for img in val_coco["images"]:
        img_id = img["id"]
        anns = anns_by_image.get(img_id, [])
        if anns:
            boxes = np.array([
                [a["bbox"][0], a["bbox"][1],
                 a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in anns
            ])
            labels = np.array([a["category_id"] for a in anns], dtype=int)
        else:
            boxes = np.zeros((0, 4))
            labels = np.zeros((0,), dtype=int)
        gt_dict[img_id] = {"boxes": boxes, "labels": labels}

    return gt_dict


@torch.no_grad()
def evaluate_competition_metric(
    model: torch.nn.Module,
    postprocess,
    val_coco: dict,
    images_dir: Path | None = None,
    resolution: int = 880,
    threshold: float = 0.2,
    device: torch.device = torch.device("cuda"),
    max_image_dim: int = 0,
) -> tuple[ScoreResult, list[dict], list[dict]]:
    """Run full-image patched inference on val set and compute competition score.

    This matches actual submission behavior: full images → overlapping patches →
    model inference → stitch with NMS → score.

    Returns:
        (ScoreResult, ground_truths, predictions) — the latter two can be passed
        to generate_analysis_plots() for detailed wandb logging.
    """
    model.eval()
    predict_fn = build_predict_fn(model, postprocess, resolution, threshold, device)
    gt_dict = _coco_to_gt(val_coco)

    ground_truths = []
    predictions = []

    for i, img_info in enumerate(val_coco["images"]):
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        # Resolve path
        img_path = Path(file_name)
        if not img_path.is_absolute() and images_dir is not None:
            img_path = images_dir / file_name

        image = Image.open(img_path).convert("RGB")

        # Downscale large images (must match training)
        from norgesgruppen.config import downscale_if_needed
        image, scale = downscale_if_needed(image, max_image_dim)

        detections = predict_with_patches(
            image, predict_fn,
            patch_size=resolution, min_overlap=resolution // 2,
        )

        # Scale predictions back to original image coordinates for scoring
        if scale != 1.0 and not detections.is_empty():
            detections = sv.Detections(
                xyxy=detections.xyxy / scale,
                confidence=detections.confidence,
                class_id=detections.class_id,
            )

        # Convert to scoring format
        if detections.is_empty():
            predictions.append({
                "boxes": np.zeros((0, 4)),
                "labels": np.zeros((0,), dtype=int),
                "scores": np.zeros((0,)),
            })
        else:
            predictions.append({
                "boxes": detections.xyxy,
                "labels": detections.class_id,
                "scores": detections.confidence,
            })

        ground_truths.append(gt_dict[img_id])

        if (i + 1) % 20 == 0:
            print(f"  Competition eval: {i + 1}/{len(val_coco['images'])} images")

    score = compute_score(ground_truths, predictions)
    return score, ground_truths, predictions


# ---------------------------------------------------------------------------
# Error-coded visualization (matches analyze_holdout_errors.py style)
# GREEN=correct, YELLOW=misclassified, RED=false positive, BLUE=missed GT
# ---------------------------------------------------------------------------

_GREEN = (0, 200, 0)
_YELLOW = (0, 220, 255)
_RED = (0, 0, 220)
_BLUE = (255, 160, 0)


def _draw_label_cv2(img, text, x, y, color, above=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.38, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if above:
        cv2.rectangle(img, (x, y - th - 4), (x + tw, y), color, -1)
        cv2.putText(img, text, (x, y - 2), font, scale, (0, 0, 0), thick)
    else:
        cv2.rectangle(img, (x, y), (x + tw, y + th + 4), color, -1)
        cv2.putText(img, text, (x, y + th + 2), font, scale, (0, 0, 0), thick)


def _match_and_classify(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """Match predictions to GT. Returns (status, gt_label, iou, missed_indices)."""
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)

    pred_status = ["fp"] * n_pred
    pred_gt_label = [None] * n_pred
    pred_iou = [0.0] * n_pred

    if n_gt == 0 or n_pred == 0:
        return pred_status, pred_gt_label, pred_iou, list(range(n_gt))

    iou_matrix = _compute_iou_matrix(pred_boxes, gt_boxes)
    gt_matched = np.zeros(n_gt, dtype=bool)

    for pred_idx in np.argsort(-pred_scores):
        ious = iou_matrix[pred_idx].copy()
        ious[gt_matched] = 0.0
        best_gt = np.argmax(ious)
        best_iou = ious[best_gt]
        pred_iou[pred_idx] = best_iou

        if best_iou >= iou_thresh:
            gt_matched[best_gt] = True
            pred_gt_label[pred_idx] = int(gt_labels[best_gt])
            if pred_labels[pred_idx] == gt_labels[best_gt]:
                pred_status[pred_idx] = "correct"
            else:
                pred_status[pred_idx] = "misclassified"

    missed_idx = [i for i in range(n_gt) if not gt_matched[i]]
    return pred_status, pred_gt_label, pred_iou, missed_idx


def _draw_error_viz(img_bgr, pred_boxes, pred_labels, pred_scores, pred_status,
                    pred_gt_label, pred_iou, missed_gt_boxes, missed_gt_labels,
                    cat_names):
    """Draw color-coded error boxes on a single image (BGR, in-place)."""
    # Missed GT (blue)
    for i in range(len(missed_gt_boxes)):
        x1, y1, x2, y2 = missed_gt_boxes[i].astype(int)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), _BLUE, 3)
        text = f"MISSED: {cat_names.get(int(missed_gt_labels[i]), '?')[:25]}"
        _draw_label_cv2(img_bgr, text, x1, y2 + 2, _BLUE, above=False)

    # Predictions
    for i in range(len(pred_boxes)):
        x1, y1, x2, y2 = pred_boxes[i].astype(int)
        status = pred_status[i]
        cat_id = pred_labels[i]
        conf = pred_scores[i]
        iou = pred_iou[i]

        if status == "correct":
            color, thickness = _GREEN, 2
            text = f"{cat_names.get(cat_id, str(cat_id))[:22]} {conf:.2f}"
        elif status == "misclassified":
            color, thickness = _YELLOW, 3
            pred_name = cat_names.get(cat_id, str(cat_id))[:18]
            gt_name = cat_names.get(pred_gt_label[i], str(pred_gt_label[i]))[:18]
            text = f"PRED:{pred_name} GT:{gt_name} iou={iou:.2f}"
        else:
            color, thickness = _RED, 2
            text = f"FP: {cat_names.get(cat_id, str(cat_id))[:20]} {conf:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
        _draw_label_cv2(img_bgr, text, x1, y1, color, above=True)

    return img_bgr


@torch.no_grad()
def render_val_predictions(
    model: torch.nn.Module,
    postprocess,
    val_coco: dict,
    images_dir: Path | None = None,
    n_images: int = 8,
    resolution: int = 880,
    threshold: float = 0.2,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
    max_image_dim: int = 0,
) -> list:
    """Render error-coded prediction overlays for a sample of val images.

    Color code: GREEN=correct, YELLOW=misclassified, RED=FP, BLUE=missed GT.
    Returns list of wandb.Image objects.
    """
    import wandb

    model.eval()
    predict_fn = build_predict_fn(model, postprocess, resolution, threshold, device)
    gt_dict = _coco_to_gt(val_coco)

    # Load category names
    cat_names = {c["id"]: c["name"] for c in val_coco.get("categories", [])}

    # Fixed deterministic sample
    rng = np.random.default_rng(seed)
    n = min(n_images, len(val_coco["images"]))
    indices = rng.choice(len(val_coco["images"]), size=n, replace=False)

    wandb_images = []

    for idx in indices:
        img_info = val_coco["images"][idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = Path(file_name)
        if not img_path.is_absolute() and images_dir is not None:
            img_path = images_dir / file_name

        image_full = Image.open(img_path).convert("RGB")

        # Downscale for inference (must match training)
        from norgesgruppen.config import downscale_if_needed
        image_infer, scale = downscale_if_needed(image_full, max_image_dim)

        detections = predict_with_patches(
            image_infer, predict_fn,
            patch_size=resolution, min_overlap=resolution // 2,
        )

        gt = gt_dict[img_id]

        # Extract arrays and scale back to original coords for visualization
        if detections.is_empty():
            pred_boxes = np.zeros((0, 4))
            pred_labels = np.zeros((0,), dtype=int)
            pred_scores = np.zeros((0,))
        else:
            pred_boxes = detections.xyxy / scale
            pred_labels = detections.class_id
            pred_scores = detections.confidence
        image = image_full

        # Match predictions to GT
        status, gt_label, iou, missed_idx = _match_and_classify(
            gt["boxes"], gt["labels"], pred_boxes, pred_labels, pred_scores,
        )

        missed_gt_boxes = gt["boxes"][missed_idx] if missed_idx else np.zeros((0, 4))
        missed_gt_labels = gt["labels"][missed_idx] if missed_idx else np.zeros((0,), dtype=int)

        # Draw on BGR image
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _draw_error_viz(
            img_bgr, pred_boxes, pred_labels, pred_scores, status,
            gt_label, iou, missed_gt_boxes, missed_gt_labels, cat_names,
        )

        # Add legend
        n_correct = sum(1 for s in status if s == "correct")
        n_miscl = sum(1 for s in status if s == "misclassified")
        n_fp = sum(1 for s in status if s == "fp")
        n_missed = len(missed_idx)
        legend = f"Correct:{n_correct} Miscl:{n_miscl} FP:{n_fp} Missed:{n_missed}"
        cv2.putText(img_bgr, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert back to PIL for wandb
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Scale down for wandb
        max_width = 2400
        if pil_img.width > max_width:
            ratio = max_width / pil_img.width
            pil_img = pil_img.resize(
                (max_width, int(pil_img.height * ratio)), Image.LANCZOS
            )

        name = Path(file_name).name
        caption = f"{name}: {n_correct}ok {n_miscl}miscl {n_fp}fp {n_missed}miss"
        wandb_images.append(wandb.Image(pil_img, caption=caption))

    return wandb_images


def generate_analysis_plots(
    ground_truths: list[dict],
    predictions: list[dict],
    val_coco: dict,
) -> dict[str, "wandb.Image"]:
    """Generate classification analysis plots for wandb logging.

    Returns dict of wandb.Image objects keyed by plot name:
        - "analysis/confusion_matrix"
        - "analysis/error_breakdown"
        - "analysis/per_class_ap"
        - "analysis/score_impact"

    Args:
        ground_truths: list of {"boxes": ndarray, "labels": ndarray}
        predictions: list of {"boxes": ndarray, "labels": ndarray, "scores": ndarray}
        val_coco: COCO dict (for category names)
    """
    import io
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import wandb

    import sys
    _repo_root = str(Path(__file__).resolve().parents[2])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from classification_analysis import (
        build_confusion_matrix,
        compute_per_class_stats,
        compute_score_impact,
        plot_ap_distribution,
        plot_confusion_matrix,
        plot_error_type_breakdown,
        plot_score_impact,
    )

    cat_names = {c["id"]: c["name"] for c in val_coco.get("categories", [])}

    per_class = compute_per_class_stats(ground_truths, predictions, cat_names)
    impacts = compute_score_impact(per_class, cat_names)
    matrix, labels, involved = build_confusion_matrix(per_class, cat_names)

    # Generate plots to a temp directory and read them back as wandb.Image
    # NOTE: use mkdtemp (not TemporaryDirectory context manager) so files
    # survive until wandb.log() has finished reading them.
    save_dir = Path(tempfile.mkdtemp())

    plot_ap_distribution(per_class, cat_names, save_dir)
    plot_score_impact(impacts, save_dir)
    plot_confusion_matrix(matrix, labels, save_dir)
    plot_error_type_breakdown(per_class, cat_names, save_dir)

    result = {}
    for name in ["per_class_ap", "score_impact", "confusion_matrix", "error_breakdown"]:
        png_path = save_dir / f"{name}.png"
        if png_path.exists():
            result[f"analysis/{name}"] = wandb.Image(str(png_path))

    return result
