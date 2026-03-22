"""Extract false positive crops from model predictions on the validation set.

Runs a trained model on the held-out validation images, compares predictions
against cleaned v2 ground truth, and saves crops of all false positive
detections (predictions with IoU < 0.5 against any GT box).

These crops can then be used as hard negatives in synthetic data generation.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from norgesgruppen.competition_eval import build_predict_fn, evaluate_competition_metric
from norgesgruppen.patching import predict_with_patches
from norgesgruppen.scoring import _compute_iou_matrix
from norgesgruppen.splitting import prepare_split_datasets
from norgesgruppen.training import _build_and_load_model, _make_args, _MODEL_ARCH


def extract_false_positives(
    checkpoint_path: str,
    output_dir: str = "data/false_positive_crops",
    model_size: str = "xxlarge",
    eval_annotations: str = "data/train/annotations.json",
    train_annotations: str = "data/dataset/train/_annotations.cleaned_v2.coco.json",
    val_fraction: float = 0.5,
    seed: int = 42,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.2,
    max_image_dim: int = 0,
):
    device = torch.device("cuda")
    arch = _MODEL_ARCH[model_size]
    resolution = arch["resolution"]
    num_classes = 357  # 356 + 1

    # Build and load model
    print(f"Loading model from {checkpoint_path}...")
    args = _make_args(
        dataset_dir=".",
        num_classes=num_classes,
        output_dir=".",
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        lr=1e-4,
        num_workers=0,
        model_size=model_size,
    )

    from rfdetr.models import build_criterion_and_postprocessors, build_model
    from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights

    model = _build_and_load_model(args, arch["pretrain_weights"])

    # Load fine-tuned weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    elif "model_ema" in ckpt:
        model.load_state_dict(ckpt["model_ema"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    # Build postprocess
    _, postprocess = build_criterion_and_postprocessors(args)
    predict_fn = build_predict_fn(model, postprocess, resolution, confidence_threshold, device)

    # Prepare split to get val images
    images_dir = Path("data/train/images")
    eval_coco_path = Path(eval_annotations)
    train_coco_path = Path(train_annotations) if train_annotations else None

    import hashlib
    from norgesgruppen.splitting import EXCLUDED_IMAGE_IDS
    ps = resolution
    mo = ps // 2
    config_str = f"{eval_coco_path}:{train_coco_path}:{val_fraction}:{seed}:{ps}:{mo}:random:1:{sorted(EXCLUDED_IMAGE_IDS)}:False:{max_image_dim}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    split_dir = Path(f"data/split-{config_hash}")

    dataset_dir, val_coco = prepare_split_datasets(
        coco_path=eval_coco_path,
        images_dir=images_dir,
        output_dir=split_dir,
        val_fraction=val_fraction,
        seed=seed,
        patch_size=ps,
        min_overlap=mo,
        crop_mode="random",
        oversample_factor=1,
        train_coco_path=train_coco_path,
        max_image_dim=max_image_dim,
    )

    if val_coco is None:
        print("ERROR: No validation set (val_fraction=0?)")
        return

    # Load cleaned v2 annotations for the val images
    # val_coco already has the eval annotations (original labels)
    # We need the cleaned v2 ground truth for accurate FP identification
    with open(train_annotations) as f:
        cleaned_coco = json.load(f)

    # Build cleaned GT indexed by image filename
    cleaned_anns_by_filename: dict[str, list[dict]] = defaultdict(list)
    cleaned_img_id_to_file = {}
    for img in cleaned_coco["images"]:
        fname = Path(img["file_name"]).name
        cleaned_img_id_to_file[img["id"]] = fname
    for ann in cleaned_coco["annotations"]:
        fname = cleaned_img_id_to_file.get(ann["image_id"], "")
        if fname:
            cleaned_anns_by_filename[fname].append(ann)

    cat_names = {c["id"]: c["name"] for c in cleaned_coco["categories"]}

    # Run inference on val images and collect FP crops
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_metadata = []
    fp_count = 0
    total_preds = 0
    total_tp = 0

    print(f"\nRunning inference on {len(val_coco['images'])} val images...")

    for i, img_info in enumerate(val_coco["images"]):
        file_name = img_info["file_name"]
        img_path = Path(file_name)
        if not img_path.is_absolute():
            img_path = images_dir / file_name

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Downscale if needed
        from norgesgruppen.config import downscale_if_needed
        image_scaled, scale = downscale_if_needed(image, max_image_dim)

        # Run patched inference
        detections = predict_with_patches(
            image_scaled, predict_fn,
            patch_size=resolution, min_overlap=resolution // 2,
        )

        if detections.is_empty():
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(val_coco['images'])} images, {fp_count} FPs so far")
            continue

        # Scale back to original coords
        pred_boxes_xyxy = detections.xyxy.copy()
        if scale != 1.0:
            pred_boxes_xyxy /= scale
        pred_labels = detections.class_id
        pred_scores = detections.confidence

        total_preds += len(pred_boxes_xyxy)

        # Get cleaned GT for this image
        img_fname = Path(file_name).name
        gt_anns = cleaned_anns_by_filename.get(img_fname, [])
        if gt_anns:
            gt_boxes_xywh = np.array([a["bbox"] for a in gt_anns])
            gt_boxes_xyxy = gt_boxes_xywh.copy()
            gt_boxes_xyxy[:, 2] += gt_boxes_xyxy[:, 0]  # x+w
            gt_boxes_xyxy[:, 3] += gt_boxes_xyxy[:, 1]  # y+h
        else:
            gt_boxes_xyxy = np.zeros((0, 4))

        # Compute IoU between predictions and GT
        if len(gt_boxes_xyxy) > 0 and len(pred_boxes_xyxy) > 0:
            iou_matrix = _compute_iou_matrix(pred_boxes_xyxy, gt_boxes_xyxy)
            max_iou_per_pred = iou_matrix.max(axis=1)
        else:
            max_iou_per_pred = np.zeros(len(pred_boxes_xyxy))

        # Load original (unscaled) image for cropping
        orig_image = Image.open(img_path).convert("RGB")

        # Extract FP crops: predictions with max IoU < threshold
        for j in range(len(pred_boxes_xyxy)):
            if max_iou_per_pred[j] < iou_threshold:
                # This is a false positive
                x1, y1, x2, y2 = pred_boxes_xyxy[j].astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w, x2)
                y2 = min(orig_h, y2)

                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                crop = orig_image.crop((x1, y1, x2, y2))
                pred_cat = int(pred_labels[j])
                pred_score = float(pred_scores[j])

                crop_fname = f"fp_{fp_count:05d}_cat{pred_cat}_score{pred_score:.2f}.jpg"
                crop.save(out_dir / crop_fname, quality=90)

                fp_metadata.append({
                    "filename": crop_fname,
                    "source_image": img_fname,
                    "predicted_category_id": pred_cat,
                    "predicted_category_name": cat_names.get(pred_cat, str(pred_cat)),
                    "confidence": pred_score,
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "max_iou_with_gt": float(max_iou_per_pred[j]),
                })
                fp_count += 1
            else:
                total_tp += 1

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(val_coco['images'])} images, {fp_count} FPs, {total_tp} TPs")

    # Save metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "total_predictions": total_preds,
            "true_positives": total_tp,
            "false_positives": fp_count,
            "checkpoint": checkpoint_path,
            "iou_threshold": iou_threshold,
            "confidence_threshold": confidence_threshold,
            "crops": fp_metadata,
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Total predictions: {total_preds}")
    print(f"True positives (IoU >= {iou_threshold}): {total_tp}")
    print(f"False positives (IoU < {iou_threshold}): {fp_count}")
    print(f"FP rate: {fp_count/max(1,total_preds)*100:.1f}%")

    # Category breakdown of FPs
    fp_cats = Counter(m["predicted_category_id"] for m in fp_metadata)
    print(f"\nTop 20 FP categories:")
    for cat_id, count in fp_cats.most_common(20):
        name = cat_names.get(cat_id, str(cat_id))
        print(f"  {count:4d} × {name}")

    print(f"\nSaved {fp_count} FP crops to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="data/false_positive_crops")
    parser.add_argument("--model-size", default="xxlarge")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--max-image-dim", type=int, default=0)
    args = parser.parse_args()

    extract_false_positives(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_size=args.model_size,
        confidence_threshold=args.confidence_threshold,
        max_image_dim=args.max_image_dim,
    )
