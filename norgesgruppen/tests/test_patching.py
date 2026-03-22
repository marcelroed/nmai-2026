"""Test that patch → predict → stitch is virtually lossless."""

import numpy as np
import supervision as sv
from PIL import Image

from norgesgruppen.patching import (
    compute_patch_grid,
    crop_boxes,
    predict_with_patches,
)


def test_patch_grid_covers_full_image():
    """Every pixel must be covered by at least one patch."""
    for img_w, img_h in [(3000, 3000), (4032, 3024), (880, 880), (500, 400), (960, 1280)]:
        patches = compute_patch_grid(img_w, img_h, patch_size=880, min_overlap=176)
        covered = np.zeros((img_h, img_w), dtype=bool)
        for px, py in patches:
            cw = min(880, img_w - px)
            ch = min(880, img_h - py)
            covered[py:py + ch, px:px + cw] = True
        assert covered.all(), f"Not all pixels covered for {img_w}x{img_h}"


def test_patch_grid_small_image():
    """Image smaller than patch_size should produce a single patch."""
    patches = compute_patch_grid(500, 400, patch_size=880)
    assert len(patches) == 1
    assert patches[0] == (0, 0)


def test_patch_grid_exact_size():
    """Image exactly patch_size should produce a single patch."""
    patches = compute_patch_grid(880, 880, patch_size=880)
    assert len(patches) == 1
    assert patches[0] == (0, 0)


def test_crop_boxes_fully_inside():
    """Boxes fully inside the crop should be preserved exactly."""
    boxes = np.array([[100, 100, 50, 50]], dtype=float)
    cats = np.array([0])
    cropped, kept_cats = crop_boxes(boxes, cats, 0, 0, 880, 880)
    assert len(cropped) == 1
    np.testing.assert_array_almost_equal(cropped[0], [100, 100, 50, 50])


def test_crop_boxes_partially_outside():
    """Box partially outside should be clipped; kept if enough is visible."""
    # Box at x=860, w=100 -> extends 80px outside an 880-wide crop
    boxes = np.array([[860, 100, 100, 100]], dtype=float)
    cats = np.array([5])
    # min_visible=0.3, original area=10000, clipped area=20*100=2000 -> 0.2 < 0.3 -> dropped
    cropped, _ = crop_boxes(boxes, cats, 0, 0, 880, 880, min_visible_fraction=0.6)
    assert len(cropped) == 0

    # With lower threshold it should be kept
    cropped, kept_cats = crop_boxes(boxes, cats, 0, 0, 880, 880, min_visible_fraction=0.1)
    assert len(cropped) == 1
    assert kept_cats[0] == 5
    np.testing.assert_array_almost_equal(cropped[0], [860, 100, 20, 100])


def test_crop_boxes_fully_outside():
    """Box fully outside the crop should be dropped."""
    boxes = np.array([[1000, 1000, 50, 50]], dtype=float)
    cats = np.array([0])
    cropped, _ = crop_boxes(boxes, cats, 0, 0, 880, 880)
    assert len(cropped) == 0


def _build_patch_predictions(img_w, img_h, patch_size, gt_boxes_xywh, gt_cats, noise_std=0.0, rng=None):
    """Helper: build per-patch detections from GT boxes, with optional noise."""
    patches = compute_patch_grid(img_w, img_h, patch_size)
    all_patch_dets = {}

    for px, py in patches:
        cw = min(patch_size, img_w - px)
        ch = min(patch_size, img_h - py)
        cropped_boxes, cropped_cats = crop_boxes(
            gt_boxes_xywh, gt_cats, px, py, cw, ch, min_visible_fraction=0.6
        )
        if len(cropped_boxes) == 0:
            all_patch_dets[(px, py)] = sv.Detections.empty()
            continue

        boxes = cropped_boxes.copy()
        if noise_std > 0 and rng is not None:
            boxes = boxes + rng.normal(0, noise_std, boxes.shape)
            boxes[:, 2:] = np.maximum(boxes[:, 2:], 5)

        xyxy = np.stack([
            boxes[:, 0], boxes[:, 1],
            boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3],
        ], axis=1)
        all_patch_dets[(px, py)] = sv.Detections(
            xyxy=xyxy, confidence=np.ones(len(xyxy)) * 0.9,
            class_id=cropped_cats.astype(int),
        )

    return patches, all_patch_dets


def _run_stitch(img_w, img_h, patch_size, patches, all_patch_dets, stitch_nms_iou=0.5):
    """Helper: stitch patch predictions back to full image."""
    image = Image.new("RGB", (img_w, img_h))
    patch_iter = iter(patches)

    def predict_fn(patch_img):
        px, py = next(patch_iter)
        return all_patch_dets[(px, py)]

    return predict_with_patches(image, predict_fn, patch_size=patch_size, stitch_nms_iou=stitch_nms_iou)


def test_roundtrip_perfect():
    """With a perfect detector (no noise), every GT box should be recovered.

    NMS may pick a clipped duplicate at patch boundaries (same confidence),
    so IoU won't always be 1.0, but should be very high.
    """
    rng = np.random.default_rng(42)
    img_w, img_h = 3000, 3000
    patch_size = 880
    n_gt = 80

    gt_x = rng.uniform(50, img_w - 100, n_gt)
    gt_y = rng.uniform(50, img_h - 100, n_gt)
    gt_w = rng.uniform(30, 150, n_gt)
    gt_h = rng.uniform(30, 150, n_gt)
    gt_boxes_xywh = np.stack([gt_x, gt_y, gt_w, gt_h], axis=1)
    gt_cats = rng.integers(0, 10, n_gt)
    gt_xyxy = np.stack([gt_x, gt_y, gt_x + gt_w, gt_y + gt_h], axis=1)

    patches, all_patch_dets = _build_patch_predictions(
        img_w, img_h, patch_size, gt_boxes_xywh, gt_cats, noise_std=0.0
    )
    result = _run_stitch(img_w, img_h, patch_size, patches, all_patch_dets)

    assert not result.is_empty()

    from norgesgruppen.scoring import _compute_iou_matrix
    iou_matrix = _compute_iou_matrix(result.xyxy, gt_xyxy)

    for gt_idx in range(n_gt):
        best_iou = iou_matrix[:, gt_idx].max()
        assert best_iou > 0.99, (
            f"GT {gt_idx} not recovered: IoU={best_iou:.3f}, box={gt_boxes_xywh[gt_idx]}"
        )

    # No duplicates
    assert len(result) == n_gt, f"Expected {n_gt} predictions, got {len(result)}"


def test_roundtrip_with_noise():
    """With detection noise, recovery should still be high but not perfect."""
    rng = np.random.default_rng(42)
    img_w, img_h = 3000, 3000
    patch_size = 880
    n_gt = 80

    gt_x = rng.uniform(50, img_w - 100, n_gt)
    gt_y = rng.uniform(50, img_h - 100, n_gt)
    gt_w = rng.uniform(30, 150, n_gt)
    gt_h = rng.uniform(30, 150, n_gt)
    gt_boxes_xywh = np.stack([gt_x, gt_y, gt_w, gt_h], axis=1)
    gt_cats = rng.integers(0, 10, n_gt)
    gt_xyxy = np.stack([gt_x, gt_y, gt_x + gt_w, gt_y + gt_h], axis=1)

    patches, all_patch_dets = _build_patch_predictions(
        img_w, img_h, patch_size, gt_boxes_xywh, gt_cats, noise_std=2.0, rng=rng
    )
    result = _run_stitch(img_w, img_h, patch_size, patches, all_patch_dets)

    assert not result.is_empty()

    from norgesgruppen.scoring import _compute_iou_matrix
    iou_matrix = _compute_iou_matrix(result.xyxy, gt_xyxy)

    gt_matched = np.array([iou_matrix[:, i].max() > 0.7 for i in range(n_gt)])
    recall = gt_matched.sum() / n_gt
    # With noise, NMS may pick a slightly worse duplicate — but recall should stay high
    assert recall > 0.95, f"Noisy roundtrip recall too low: {recall:.2f} ({gt_matched.sum()}/{n_gt})"


def test_roundtrip_boundary_boxes():
    """Boxes deliberately placed on patch boundaries should survive stitching."""
    img_w, img_h = 2000, 2000
    patch_size = 880

    patches = compute_patch_grid(img_w, img_h, patch_size)
    # Place boxes exactly on boundaries between patches
    boundary_boxes = []
    for i in range(len(patches) - 1):
        px1, py1 = patches[i]
        px2, py2 = patches[min(i + 1, len(patches) - 1)]
        if px2 > px1:  # horizontal boundary
            bx = px1 + patch_size - 20  # straddles 20px into next patch
            by = py1 + 100
            boundary_boxes.append([bx, by, 60, 60])
        if py2 > py1:  # vertical boundary
            bx = px1 + 100
            by = py1 + patch_size - 20
            boundary_boxes.append([bx, by, 60, 60])

    if not boundary_boxes:
        return

    gt_boxes = np.array(boundary_boxes, dtype=float)
    gt_cats = np.zeros(len(gt_boxes), dtype=int)
    gt_xyxy = np.stack([
        gt_boxes[:, 0], gt_boxes[:, 1],
        gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3],
    ], axis=1)

    # Build per-patch detections from GT
    all_patch_dets = {}
    for px, py in patches:
        cw = min(patch_size, img_w - px)
        ch = min(patch_size, img_h - py)
        cropped, ccats = crop_boxes(gt_boxes, gt_cats, px, py, cw, ch, min_visible_fraction=0.6)
        if len(cropped) == 0:
            all_patch_dets[(px, py)] = sv.Detections.empty()
            continue
        xyxy = np.stack([
            cropped[:, 0], cropped[:, 1],
            cropped[:, 0] + cropped[:, 2], cropped[:, 1] + cropped[:, 3],
        ], axis=1)
        all_patch_dets[(px, py)] = sv.Detections(
            xyxy=xyxy,
            confidence=np.ones(len(xyxy)) * 0.9,
            class_id=ccats.astype(int),
        )

    image = Image.new("RGB", (img_w, img_h))
    patch_iter = iter(patches)

    def predict_fn(patch_img):
        px, py = next(patch_iter)
        return all_patch_dets[(px, py)]

    result = predict_with_patches(image, predict_fn, patch_size=patch_size, stitch_nms_iou=0.5)

    from norgesgruppen.scoring import _compute_iou_matrix
    iou_matrix = _compute_iou_matrix(result.xyxy, gt_xyxy)

    for gt_idx in range(len(gt_boxes)):
        best_iou = iou_matrix[:, gt_idx].max() if len(iou_matrix) > 0 else 0
        assert best_iou > 0.7, (
            f"Boundary box {gt_idx} not recovered: best IoU={best_iou:.2f}, "
            f"box={gt_boxes[gt_idx]}"
        )
