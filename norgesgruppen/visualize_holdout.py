#!/usr/bin/env python3
"""Predict and visualize bounding boxes on holdout images using a trained checkpoint."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from norgesgruppen.config import MODEL_CLS, NUM_CLASSES
from norgesgruppen.patching import predict_with_patches
from norgesgruppen.postprocess import DEFAULT_TRANSFORMS, apply_transforms

# ---------------------------------------------------------------------------
# Holdout images
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


def load_category_names() -> dict[int, str]:
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    return {cat["id"]: cat["name"] for cat in data["categories"]}


def load_model(checkpoint_path: str):
    """Load RF-DETR model from checkpoint."""
    model = MODEL_CLS()
    model.model.reinitialize_detection_head(num_classes=NUM_CLASSES + 1)

    import argparse as _argparse
    torch.serialization.add_safe_globals([_argparse.Namespace])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.model.model.load_state_dict(ckpt[state_key])
    print(f"Loaded {state_key} weights (epoch {ckpt.get('epoch', '?')})")

    model.model.model.eval()
    model.model.model.cuda()
    return model


def predict_single(model, image: Image.Image, threshold: float = 0.2):
    """Run patch-based inference on a single image."""
    detections = predict_with_patches(
        image, lambda img: model.predict(img, threshold=threshold)
    )
    return apply_transforms(detections, DEFAULT_TRANSFORMS)


def draw_detections(image_path: Path, detections, category_names: dict, conf_threshold: float):
    """Draw detection boxes on image and return BGR numpy array."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return None

    rng = np.random.default_rng(42)
    color_map: dict[int, tuple] = {}

    for i in range(len(detections)):
        conf = detections.confidence[i]
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        cat_id = detections.class_id[i]

        if cat_id not in color_map:
            color_map[cat_id] = tuple(int(c) for c in rng.integers(80, 255, 3))
        color = color_map[cat_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = category_names.get(cat_id, str(cat_id))
        label_text = f"{label[:25]} {conf:.2f}"
        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_1e-3_train_split.pth")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for visualization")
    parser.add_argument("--predict-threshold", type=float, default=0.2, help="Model prediction threshold")
    parser.add_argument("--save-dir", type=Path, default=Path("viz_holdout"))
    args = parser.parse_args()

    category_names = load_category_names()
    print(f"Loaded {len(category_names)} category names")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    for img_name in HOLDOUT_IMAGES:
        image_path = IMAGES_DIR / img_name
        if not image_path.exists():
            print(f"  SKIP (not found): {image_path}")
            continue

        print(f"  Predicting: {img_name}...", end=" ", flush=True)
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad(), torch.amp.autocast("cuda"):
            detections = predict_single(model, image, threshold=args.predict_threshold)

        n_total = len(detections)
        n_shown = int((detections.confidence >= args.conf).sum()) if n_total > 0 else 0
        print(f"{n_total} detections, {n_shown} shown (conf >= {args.conf})")

        viz = draw_detections(image_path, detections, category_names, args.conf)
        if viz is not None:
            stem = Path(img_name).stem
            out_path = args.save_dir / f"{stem}_pred.jpg"
            cv2.imwrite(str(out_path), viz)

    print(f"\nVisualizations saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
