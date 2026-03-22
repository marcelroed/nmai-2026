#!/usr/bin/env python3
"""Visualize bounding box predictions from Run Output Testing.json."""

import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGES_DIR = Path(__file__).parent / "data/train/images"
JSON_PATH = Path(__file__).parent / "src/norgesgruppen/Run Output Testing.json"


def load_category_names(annotations_path: Path) -> dict[int, str]:
    with open(annotations_path) as f:
        data = json.load(f)
    return {cat["id"]: cat["name"] for cat in data.get("categories", [])}


def draw_predictions(image_path: Path, predictions: list[dict], category_names: dict, conf_threshold: float = 0.5):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return None

    rng = np.random.default_rng(42)
    color_map: dict[int, tuple] = {}

    for pred in predictions:
        conf = pred["confidence"]
        if conf < conf_threshold:
            continue

        x, y, w, h = pred["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cat_id = pred["category_id"]

        if cat_id not in color_map:
            color_map[cat_id] = tuple(int(c) for c in rng.integers(80, 255, 3))
        color = color_map[cat_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = category_names.get(cat_id, str(cat_id))
        label_text = f"{label[:20]} {conf:.2f}"
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions with bounding boxes.")
    parser.add_argument("--n", type=int, default=5, help="Number of images to show (default: 5)")
    parser.add_argument("--image-ids", nargs="+", help="Specific image IDs to show (e.g. img_00001)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Save images to directory instead of displaying")
    args = parser.parse_args()

    print(f"Loading predictions from {JSON_PATH}...")
    with open(JSON_PATH) as f:
        all_predictions = json.load(f)

    annotations_path = Path(__file__).parent / "data/dataset/train/_annotations.coco.json"
    category_names = {}
    if annotations_path.exists():
        category_names = load_category_names(annotations_path)
        print(f"Loaded {len(category_names)} category names.")

    pred_by_id = {entry["image_id"]: entry["predictions"] for entry in all_predictions}

    if args.image_ids:
        selected = [i for i in args.image_ids if i in pred_by_id]
    else:
        available = [k for k in pred_by_id if (IMAGES_DIR / f"{k}.jpg").exists()]
        selected = random.sample(available, min(args.n, len(available)))

    print(f"Visualizing {len(selected)} images (conf >= {args.conf})...")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for image_id in selected:
        image_path = IMAGES_DIR / f"{image_id}.jpg"
        predictions = pred_by_id.get(image_id, [])
        img = draw_predictions(image_path, predictions, category_names, conf_threshold=args.conf)
        if img is None:
            continue

        n_shown = sum(1 for p in predictions if p["confidence"] >= args.conf)
        print(f"  {image_id}: {n_shown}/{len(predictions)} boxes shown")

        if args.save_dir:
            out_path = args.save_dir / f"{image_id}_pred.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"    Saved to {out_path}")
        else:
            cv2.imshow(image_id, img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break


if __name__ == "__main__":
    main()
