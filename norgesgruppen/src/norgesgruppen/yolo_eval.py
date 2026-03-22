"""YOLO competition evaluation bridge.

Connects YOLO inference to the existing scoring and patching pipeline
so that competition metrics are directly comparable to RF-DETR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import supervision as sv
import torch
from PIL import Image

from norgesgruppen.competition_eval import _coco_to_gt
from norgesgruppen.config import downscale_if_needed
from norgesgruppen.patching import predict_with_patches
from norgesgruppen.scoring import ScoreResult, compute_score


def build_yolo_predict_fn(
    model,
    imgsz: int = 880,
    threshold: float = 0.2,
    yolo_to_coco_map: dict[int, int] | None = None,
) -> Callable[[Image.Image], sv.Detections]:
    """Create a predict function compatible with predict_with_patches().

    Args:
        model: Ultralytics YOLO model.
        imgsz: Input image size for YOLO.
        threshold: Confidence threshold.
        yolo_to_coco_map: Mapping from YOLO contiguous indices to COCO
            category IDs. If None, class IDs are passed through unchanged.

    Returns:
        Callable that takes a PIL Image and returns sv.Detections.
    """

    def predict_fn(image: Image.Image) -> sv.Detections:
        results = model.predict(
            image,
            imgsz=imgsz,
            conf=threshold,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return sv.Detections.empty()

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        # Remap YOLO contiguous indices back to COCO category IDs
        if yolo_to_coco_map is not None:
            cls = np.array([yolo_to_coco_map[c] for c in cls])

        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)

    return predict_fn


@torch.no_grad()
def evaluate_yolo_competition(
    model,
    val_coco: dict,
    images_dir: Path,
    imgsz: int = 880,
    yolo_to_coco_map: dict[int, int] | None = None,
    max_image_dim: int = 0,
    threshold: float = 0.2,
) -> tuple[ScoreResult, list[dict], list[dict]]:
    """Run full-image patched inference with YOLO and compute competition score.

    Mirrors competition_eval.evaluate_competition_metric() but uses YOLO
    instead of RF-DETR.

    Returns:
        (ScoreResult, ground_truths, predictions)
    """
    predict_fn = build_yolo_predict_fn(model, imgsz, threshold, yolo_to_coco_map)
    gt_dict = _coco_to_gt(val_coco)

    ground_truths = []
    predictions = []

    for i, img_info in enumerate(val_coco["images"]):
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = Path(file_name)
        if not img_path.is_absolute() and images_dir is not None:
            img_path = images_dir / file_name

        image = Image.open(img_path).convert("RGB")
        image, scale = downscale_if_needed(image, max_image_dim)

        detections = predict_with_patches(
            image,
            predict_fn,
            patch_size=imgsz,
            min_overlap=imgsz // 2,
        )

        # Scale predictions back to original image coordinates
        if scale != 1.0 and not detections.is_empty():
            detections = sv.Detections(
                xyxy=detections.xyxy / scale,
                confidence=detections.confidence,
                class_id=detections.class_id,
            )

        if detections.is_empty():
            predictions.append(
                {
                    "boxes": np.zeros((0, 4)),
                    "labels": np.zeros((0,), dtype=int),
                    "scores": np.zeros((0,)),
                }
            )
        else:
            predictions.append(
                {
                    "boxes": detections.xyxy,
                    "labels": detections.class_id,
                    "scores": detections.confidence,
                }
            )

        ground_truths.append(gt_dict[img_id])

        if (i + 1) % 20 == 0:
            print(f"  YOLO competition eval: {i + 1}/{len(val_coco['images'])} images")

    score = compute_score(ground_truths, predictions)
    return score, ground_truths, predictions


def make_competition_eval_callback(
    val_coco: dict,
    images_dir: Path,
    imgsz: int,
    yolo_to_coco_map: dict[int, int],
    eval_interval: int,
    max_image_dim: int,
    output_dir: Path,
):
    """Create an Ultralytics callback for periodic competition evaluation.

    Register with: model.add_callback("on_fit_epoch_end", callback)

    Logs eval/hybrid_score, eval/detection_map, eval/classification_map
    to W&B and saves the best competition checkpoint.
    """
    best_score = [0.0]

    def callback(trainer):
        epoch = trainer.epoch
        total_epochs = trainer.epochs

        # Run at eval_interval or on the last epoch
        if (epoch + 1) % eval_interval != 0 and epoch != total_epochs - 1:
            return

        print(f"\n--- Competition eval at epoch {epoch + 1} ---")

        # Load the latest saved checkpoint as a proper YOLO model.
        # This ensures all preprocessing/postprocessing is correctly configured,
        # unlike manually assigning trainer.ema.ema to a bare YOLO() wrapper.
        from ultralytics import YOLO

        ckpt_path = trainer.last  # path to last saved .pt
        if ckpt_path is None or not Path(ckpt_path).exists():
            print("  Skipping competition eval: no checkpoint saved yet")
            return

        eval_model = YOLO(ckpt_path)

        score, gts, preds = evaluate_yolo_competition(
            eval_model,
            val_coco,
            images_dir,
            imgsz=imgsz,
            yolo_to_coco_map=yolo_to_coco_map,
            max_image_dim=max_image_dim,
        )

        print(f"  {score}")

        # Log to W&B
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/hybrid_score": score.combined,
                        "eval/detection_map": score.detection_map,
                        "eval/classification_map": score.classification_map,
                    },
                )
        except ImportError:
            pass

        # Save best checkpoint
        if score.combined > best_score[0]:
            best_score[0] = score.combined
            save_path = Path(output_dir) / "best_competition.pt"
            import shutil

            shutil.copy2(ckpt_path, save_path)
            print(f"  New best competition score: {score.combined:.4f} → {save_path}")

    return callback
