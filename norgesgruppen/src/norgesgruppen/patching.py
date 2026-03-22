"""Patch-based training and inference for large images.

Three components:
1. Tiling geometry (shared)
2. Training patch export: COCO + full images → patched COCO dataset
3. Inference stitch: full image → patches → predict → stitch back

Toggle via USE_PATCHES flag in the calling module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

PATCH_SIZE = 880
MIN_OVERLAP = 400  # pixels — must be >= largest box dimension to guarantee full containment
MIN_VISIBLE_FRACTION = 0.6  # drop boxes with less than 60% visible after crop
STITCH_NMS_IOU = 0.6


# ---------------------------------------------------------------------------
# 1. Tiling geometry
# ---------------------------------------------------------------------------


def compute_patch_grid(
    image_w: int,
    image_h: int,
    patch_size: int = PATCH_SIZE,
    min_overlap: int = MIN_OVERLAP,
) -> list[tuple[int, int]]:
    """Compute top-left (x, y) positions for overlapping patches covering the image.

    Patches are distributed evenly so overlap is uniform.
    Images smaller than patch_size in a dimension get a single patch at 0.

    Args:
        min_overlap: Minimum overlap in pixels between adjacent patches.
            Set this >= the largest expected box dimension to guarantee
            every object is fully contained in at least one patch.
    """
    def _positions(length: int) -> list[int]:
        if length <= patch_size:
            return [0]
        min_overlap_px = min(min_overlap, patch_size - 1)
        stride = patch_size - min_overlap_px
        n = max(1, int(np.ceil((length - patch_size) / stride)) + 1)
        # Distribute evenly
        if n == 1:
            return [0]
        positions = np.linspace(0, length - patch_size, n).astype(int).tolist()
        return positions

    xs = _positions(image_w)
    ys = _positions(image_h)
    return [(x, y) for y in ys for x in xs]


def crop_boxes(
    boxes_xywh: np.ndarray,
    category_ids: np.ndarray,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    min_visible_fraction: float = MIN_VISIBLE_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip COCO boxes [x, y, w, h] to a crop region, returning crop-local coords.

    Drops boxes where visible area < min_visible_fraction of original area.
    Returns (clipped_boxes_xywh, kept_category_ids).
    """
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)

    boxes = boxes_xywh.copy().astype(float)
    # Convert to xyxy
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    orig_area = boxes[:, 2] * boxes[:, 3]

    # Clip to crop
    cx1 = np.clip(x1, crop_x, crop_x + crop_w)
    cy1 = np.clip(y1, crop_y, crop_y + crop_h)
    cx2 = np.clip(x2, crop_x, crop_x + crop_w)
    cy2 = np.clip(y2, crop_y, crop_y + crop_h)

    clipped_w = cx2 - cx1
    clipped_h = cy2 - cy1
    clipped_area = clipped_w * clipped_h

    # Filter by visible fraction
    with np.errstate(divide="ignore", invalid="ignore"):
        visible = np.where(orig_area > 0, clipped_area / orig_area, 0)
    keep = (visible >= min_visible_fraction) & (clipped_w > 0) & (clipped_h > 0)

    # Convert to crop-local xywh
    result = np.stack([
        cx1[keep] - crop_x,
        cy1[keep] - crop_y,
        clipped_w[keep],
        clipped_h[keep],
    ], axis=1)

    return result, category_ids[keep]


# ---------------------------------------------------------------------------
# 2. Training patch export
# ---------------------------------------------------------------------------


def generate_patched_coco(
    coco: dict,
    output_dir: Path,
    patch_size: int = PATCH_SIZE,
    min_overlap: int = MIN_OVERLAP,
    min_visible_fraction: float = MIN_VISIBLE_FRACTION,
    subdir: str = "train",
    create_dummy_valid: bool = True,
    max_image_dim: int = 0,
) -> dict:
    """Generate a patched COCO dataset from a full-image COCO dict.

    Crops images into overlapping patches, adjusts annotations, writes
    patch images to output_dir/{subdir}/ and returns the new COCO dict.
    """
    images_dir = output_dir / subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    # Index annotations by image_id
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    next_image_id = 1
    next_ann_id = 1

    from tqdm import tqdm

    for img_info in tqdm(coco["images"], desc="Generating patches"):
        image_id = img_info["id"]
        image_path = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        image = Image.open(image_path)

        # Downscale large images before patching
        from norgesgruppen.config import downscale_if_needed
        image, scale = downscale_if_needed(image, max_image_dim)
        if scale != 1.0:
            img_w = int(img_w * scale)
            img_h = int(img_h * scale)

        anns = anns_by_image.get(image_id, [])

        if anns:
            boxes_xywh = np.array([a["bbox"] for a in anns]) * scale
            cat_ids = np.array([a["category_id"] for a in anns])
        else:
            boxes_xywh = np.zeros((0, 4))
            cat_ids = np.zeros((0,), dtype=int)

        patches = compute_patch_grid(img_w, img_h, patch_size, min_overlap)

        for px, py in patches:
            # Actual crop dimensions (handle image edge)
            cw = min(patch_size, img_w - px)
            ch = min(patch_size, img_h - py)

            # Crop image
            patch_img = image.crop((px, py, px + cw, py + ch))

            # Pad if smaller than patch_size
            if cw < patch_size or ch < patch_size:
                padded = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
                padded.paste(patch_img, (0, 0))
                patch_img = padded

            # Save patch
            patch_filename = f"patch_{next_image_id:06d}.jpg"
            patch_path = images_dir / patch_filename
            patch_img.save(patch_path, quality=95)

            # Adjust annotations
            cropped_boxes, cropped_cats = crop_boxes(
                boxes_xywh, cat_ids, px, py, cw, ch, min_visible_fraction
            )

            new_images.append({
                "id": next_image_id,
                "file_name": str(patch_path.resolve()),
                "width": patch_size,
                "height": patch_size,
            })

            for j in range(len(cropped_boxes)):
                new_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": int(cropped_cats[j]),
                    "bbox": cropped_boxes[j].tolist(),
                    "area": float(cropped_boxes[j][2] * cropped_boxes[j][3]),
                    "iscrowd": 0,
                })
                next_ann_id += 1

            next_image_id += 1

    patched_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_annotations,
    }

    # Write annotations
    with open(output_dir / subdir / "_annotations.coco.json", "w") as f:
        json.dump(patched_coco, f)

    # Dummy valid split (rfdetr requires a valid/ directory)
    if create_dummy_valid:
        (output_dir / "valid").mkdir(parents=True, exist_ok=True)
        first_image = new_images[0]
        valid_coco = {
            "info": patched_coco.get("info", {}),
            "licenses": patched_coco.get("licenses", []),
            "categories": patched_coco["categories"],
            "images": [first_image],
            "annotations": [a for a in new_annotations if a["image_id"] == first_image["id"]],
        }
        with open(output_dir / "valid" / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)

    print(f"Patched dataset: {len(new_images)} patches, {len(new_annotations)} annotations "
          f"(from {len(coco['images'])} images, {len(coco['annotations'])} annotations)")

    return patched_coco


# ---------------------------------------------------------------------------
# 3. Inference stitch
# ---------------------------------------------------------------------------


def predict_with_patches(
    image: Image.Image,
    predict_fn,
    patch_size: int = PATCH_SIZE,
    min_overlap: int = MIN_OVERLAP,
    stitch_nms_iou: float = STITCH_NMS_IOU,
) -> sv.Detections:
    """Run detection on overlapping patches and stitch results back.

    Args:
        image: Full-size PIL image.
        predict_fn: Callable(PIL.Image) -> sv.Detections
        patch_size: Size of each square patch.
        min_overlap: Minimum overlap in pixels between adjacent patches.
        stitch_nms_iou: IoU threshold for NMS when merging patch results.

    Returns:
        Merged sv.Detections in full-image coordinates.
    """
    img_w, img_h = image.size
    patches = compute_patch_grid(img_w, img_h, patch_size, min_overlap)

    all_xyxy = []
    all_confidence = []
    all_class_id = []

    for px, py in patches:
        cw = min(patch_size, img_w - px)
        ch = min(patch_size, img_h - py)

        patch_img = image.crop((px, py, px + cw, py + ch))

        # Pad if needed
        if cw < patch_size or ch < patch_size:
            padded = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
            padded.paste(patch_img, (0, 0))
            patch_img = padded

        import torch
        with torch.amp.autocast("cuda"):
            dets = predict_fn(patch_img)
        if dets.is_empty():
            continue

        # Offset to full-image coordinates
        xyxy = dets.xyxy.copy()
        xyxy[:, 0] += px
        xyxy[:, 1] += py
        xyxy[:, 2] += px
        xyxy[:, 3] += py

        # Clip to actual image bounds (discard detections on padding)
        valid = (
            (xyxy[:, 0] < img_w) & (xyxy[:, 1] < img_h) &
            (xyxy[:, 2] > 0) & (xyxy[:, 3] > 0)
        )
        xyxy = xyxy[valid]
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, img_w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, img_h)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, img_w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, img_h)

        # Drop boxes that touch a non-image patch edge — these are likely
        # truncated predictions. The same object should be fully detected in
        # a neighboring patch. We only suppress edges that are interior
        # (not at the image boundary), since a box touching the image edge
        # is legitimate.
        local_xyxy = dets.xyxy[valid]
        edge_tol = 3  # pixels
        touches_left   = (local_xyxy[:, 0] < edge_tol) & (px > 0)
        touches_top    = (local_xyxy[:, 1] < edge_tol) & (py > 0)
        touches_right  = (local_xyxy[:, 2] > cw - edge_tol) & (px + cw < img_w)
        touches_bottom = (local_xyxy[:, 3] > ch - edge_tol) & (py + ch < img_h)
        not_truncated = ~(touches_left | touches_top | touches_right | touches_bottom)

        xyxy = xyxy[not_truncated]
        local_xyxy = local_xyxy[not_truncated]
        confidence = dets.confidence[valid][not_truncated]
        class_id = dets.class_id[valid][not_truncated]

        if len(xyxy) == 0:
            continue

        # Small confidence boost for boxes far from edges so NMS prefers them.
        margin = np.min([
            local_xyxy[:, 0],
            local_xyxy[:, 1],
            cw - local_xyxy[:, 2],
            ch - local_xyxy[:, 3],
        ], axis=0)
        edge_bonus = np.clip(margin / patch_size, 0, 0.01)
        confidence = confidence + edge_bonus

        all_xyxy.append(xyxy)
        all_confidence.append(confidence)
        all_class_id.append(class_id)

    if not all_xyxy:
        return sv.Detections.empty()

    merged = sv.Detections(
        xyxy=np.concatenate(all_xyxy),
        confidence=np.concatenate(all_confidence),
        class_id=np.concatenate(all_class_id),
    )

    # NMS to deduplicate detections from overlapping patches
    if len(merged) > 0:
        merged = merged.with_nms(threshold=stitch_nms_iou, class_agnostic=True)

    return merged
