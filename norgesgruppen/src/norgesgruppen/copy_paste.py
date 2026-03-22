"""Copy-paste augmentation for object detection training.

Extracts object instances from training images and pastes them into random
crops during training, biased toward confused category counterparts and
rare categories. This forces the model to learn fine-grained differences
between visually similar products.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from norgesgruppen.splitting import CONFUSION_PAIRS


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two xywh boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _max_iou_with_existing(
    new_box: np.ndarray, existing_boxes: np.ndarray
) -> float:
    """Max IoU of new_box against all existing boxes (both xywh)."""
    if len(existing_boxes) == 0:
        return 0.0
    return max(_compute_iou(new_box, eb) for eb in existing_boxes)


def _gaussian_feather_mask(h: int, w: int, feather_px: int = 3) -> np.ndarray:
    """Create an alpha mask that feathers from 1.0 in center to 0.0 at edges."""
    mask = np.ones((h, w), dtype=np.float32)
    if feather_px <= 0 or h < 2 * feather_px or w < 2 * feather_px:
        return mask

    for i in range(feather_px):
        alpha = (i + 1) / (feather_px + 1)
        mask[i, :] = np.minimum(mask[i, :], alpha)
        mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
        mask[:, i] = np.minimum(mask[:, i], alpha)
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

    return mask


class InstanceBank:
    """Pre-extracted object instances from training images, grouped by category.

    Built once at dataset init. Each instance is a small numpy crop (uint8)
    of the bounding box region from the original full-size image.
    """

    def __init__(self, coco: dict, margin: int = 3):
        """Extract all instances from the COCO dict.

        Args:
            coco: COCO-format dict with images (file_name must be absolute
                  or relative to cwd) and annotations.
            margin: Extra pixels around each bbox to include.
        """
        self.instances: dict[int, list[np.ndarray]] = defaultdict(list)
        self.margin = margin

        # Index annotations by image_id
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        # Build confusion pair lookups
        self.confusion_counterparts: dict[int, list[int]] = defaultdict(list)
        for a, b in CONFUSION_PAIRS:
            self.confusion_counterparts[a].append(b)
            self.confusion_counterparts[b].append(a)

        # Compute per-category frequency for rare-category sampling
        cat_counts = Counter(a["category_id"] for a in coco["annotations"])
        sorted_cats = sorted(cat_counts.keys(), key=lambda c: cat_counts[c])
        n_rare = max(1, len(sorted_cats) * 30 // 100)
        self.rare_categories = sorted_cats[:n_rare]

        # Extract instances
        img_by_id = {img["id"]: img for img in coco["images"]}
        n_extracted = 0
        n_skipped = 0

        for img_id, anns in anns_by_image.items():
            img_info = img_by_id.get(img_id)
            if img_info is None:
                continue

            img_path = img_info["file_name"]
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue

            ih, iw = image.shape[:2]

            for ann in anns:
                x, y, w, h = ann["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                if w < 4 or h < 4:
                    n_skipped += 1
                    continue

                # Add margin, clamped to image bounds
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(iw, x + w + margin)
                y2 = min(ih, y + h + margin)

                crop = image[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    n_skipped += 1
                    continue

                self.instances[ann["category_id"]].append(crop)
                n_extracted += 1

        self.all_cat_ids = list(self.instances.keys())
        print(
            f"InstanceBank: extracted {n_extracted} instances "
            f"across {len(self.all_cat_ids)} categories "
            f"(skipped {n_skipped} tiny/invalid)"
        )

    def sample(
        self, cat_id: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        """Sample a random instance for the given category."""
        instances = self.instances.get(cat_id)
        if not instances:
            return None
        return instances[rng.integers(len(instances))]

    def sample_rare(self, rng: np.random.Generator) -> tuple[int, np.ndarray] | None:
        """Sample a random instance from a rare category."""
        if not self.rare_categories:
            return None
        cat_id = self.rare_categories[rng.integers(len(self.rare_categories))]
        inst = self.sample(cat_id, rng)
        if inst is None:
            return None
        return cat_id, inst

    def get_confused_counterpart(
        self, cat_id: int, rng: np.random.Generator
    ) -> tuple[int, np.ndarray] | None:
        """Get an instance of the confused counterpart for this category."""
        counterparts = self.confusion_counterparts.get(cat_id)
        if not counterparts:
            return None
        target_cat = counterparts[rng.integers(len(counterparts))]
        inst = self.sample(target_cat, rng)
        if inst is None:
            return None
        return target_cat, inst


def apply_copy_paste(
    crop_img: np.ndarray,
    boxes_xywh: np.ndarray,
    cat_ids: np.ndarray,
    instance_bank: InstanceBank,
    rng: np.random.Generator,
    n_paste: int = 3,
    p_paste: float = 0.5,
    p_confused: float = 0.5,
    resolution: int = 880,
    max_iou_overlap: float = 0.3,
    scale_range: tuple[float, float] = (0.7, 1.3),
    feather_px: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paste additional object instances into a training crop.

    Args:
        crop_img: [H, W, 3] uint8 image crop.
        boxes_xywh: [N, 4] existing annotations in xywh format.
        cat_ids: [N] category IDs for existing annotations.
        instance_bank: Pre-extracted instances.
        rng: Random number generator.
        n_paste: Number of instances to attempt pasting.
        p_paste: Probability of applying copy-paste at all.
        p_confused: Probability that each paste targets a confused counterpart.
        resolution: Crop resolution (for bounds checking).
        max_iou_overlap: Max allowed IoU with existing boxes for placement.
        scale_range: Random scale range for pasted instances.
        feather_px: Pixels of Gaussian feathering at paste edges.

    Returns:
        (modified_img, modified_boxes, modified_cats) — same formats as input.
    """
    if rng.random() > p_paste:
        return crop_img, boxes_xywh, cat_ids

    img = crop_img.copy()
    new_boxes = list(boxes_xywh) if len(boxes_xywh) > 0 else []
    new_cats = list(cat_ids) if len(cat_ids) > 0 else []

    # Collect existing box array for IoU checks
    existing = np.array(new_boxes) if new_boxes else np.zeros((0, 4))

    for _ in range(n_paste):
        # Decide what to paste
        paste_result = None

        if rng.random() < p_confused and len(new_cats) > 0:
            # Try to find a confused counterpart for a category in this crop
            crop_cats = list(set(int(c) for c in new_cats))
            rng.shuffle(crop_cats)
            for cc in crop_cats:
                result = instance_bank.get_confused_counterpart(cc, rng)
                if result is not None:
                    paste_result = result
                    break

        if paste_result is None:
            # Fall back to a rare category
            paste_result = instance_bank.sample_rare(rng)

        if paste_result is None:
            continue

        paste_cat, paste_crop = paste_result

        # Scale the instance
        scale = rng.uniform(*scale_range)
        new_h = max(4, int(round(paste_crop.shape[0] * scale)))
        new_w = max(4, int(round(paste_crop.shape[1] * scale)))

        # Clamp to fit within crop
        new_h = min(new_h, resolution - 2)
        new_w = min(new_w, resolution - 2)

        if new_h < 4 or new_w < 4:
            continue

        resized = cv2.resize(paste_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Find a valid paste position (up to 10 attempts)
        placed = False
        for _ in range(10):
            px = int(rng.integers(0, max(1, resolution - new_w)))
            py = int(rng.integers(0, max(1, resolution - new_h)))
            candidate_box = np.array([px, py, new_w, new_h], dtype=np.float64)

            if _max_iou_with_existing(candidate_box, existing) < max_iou_overlap:
                # Paste with feathered alpha blending
                alpha = _gaussian_feather_mask(new_h, new_w, feather_px)
                alpha_3d = alpha[:, :, np.newaxis]

                region = img[py : py + new_h, px : px + new_w].astype(np.float32)
                pasted = resized.astype(np.float32)
                blended = region * (1 - alpha_3d) + pasted * alpha_3d
                img[py : py + new_h, px : px + new_w] = blended.astype(np.uint8)

                # Add annotation
                new_boxes.append(candidate_box)
                new_cats.append(paste_cat)
                existing = np.array(new_boxes)
                placed = True
                break

        # If not placed after 10 attempts, skip this paste

    out_boxes = np.array(new_boxes, dtype=np.float64) if new_boxes else np.zeros((0, 4), dtype=np.float64)
    out_cats = np.array(new_cats, dtype=np.int64) if new_cats else np.zeros((0,), dtype=np.int64)

    return img, out_boxes, out_cats
