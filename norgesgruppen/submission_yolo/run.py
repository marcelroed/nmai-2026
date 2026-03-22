"""Competition submission entry point — YOLOv26 (ONNX, end2end).

Runs YOLOv26 exported to ONNX with batched patch-based inference on
full-resolution shelf images. The end2end ONNX model outputs clean
detections (no per-patch NMS needed), but cross-patch NMS is applied
when stitching overlapping patches.

Contract:
    python run.py --input /data/images --output /output/predictions.json

Sandbox: Python 3.11, onnxruntime-gpu 1.20.0, PyTorch 2.6.0+cu124,
         NVIDIA L4 24GB, no network, 300s timeout.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision

# --- Config (loaded from config.json if present, otherwise defaults) ---
_config_path = Path(__file__).parent / "config.json"
if _config_path.exists():
    with open(str(_config_path)) as _f:
        _config = json.load(_f)
else:
    _config = {}

IMGSZ = _config.get("imgsz", 640)
USE_HALF = _config.get("half", True)

# --- Inference config ---
CONFIDENCE_THRESHOLD = 0.3
BATCH_SIZE = 16

# --- Patching config ---
PATCH_SIZE = IMGSZ
MIN_OVERLAP = IMGSZ // 2
EDGE_TOLERANCE = 3
STITCH_NMS_IOU = 0.6

# --- Image downscaling (must match training --max-image-dim) ---
MAX_IMAGE_DIM = 0  # 0 = no limit

# --- Class mapping (YOLO contiguous → COCO category_id) ---
_mapping_path = Path(__file__).parent / "class_mapping.json"
if _mapping_path.exists():
    with open(str(_mapping_path)) as _f:
        _mapping = json.load(_f)
    YOLO_TO_COCO = {int(k): int(v) for k, v in _mapping["yolo_to_coco"].items()}
else:
    YOLO_TO_COCO = None  # pass-through


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def load_model(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    active = session.get_providers()
    print(f"ORT providers: {active}")
    return session


def _has_cuda_provider(session: ort.InferenceSession) -> bool:
    return "CUDAExecutionProvider" in session.get_providers()


def preprocess_batch(patches: list[np.ndarray], use_gpu: bool = True) -> np.ndarray:
    """Preprocess patches. Returns [N, 3, H, W] float array (0-1)."""
    resized = []
    for p in patches:
        if p.shape[0] != IMGSZ or p.shape[1] != IMGSZ:
            p = cv2.resize(p, (IMGSZ, IMGSZ))
        resized.append(p)
    # HWC uint8 → float32, normalize to [0, 1]
    # Always float32 for ONNX input (model handles FP16 internally)
    batch = np.stack(resized).astype(np.float32) / 255.0
    batch = batch.transpose(0, 3, 1, 2)  # NHWC → NCHW
    return np.ascontiguousarray(batch)


def decode_end2end(
    output: np.ndarray,
    n_real: int,
    patch_sizes: list[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode end2end YOLO output [B, 300, 6] → (xyxy, cids, scores) per image.

    Each detection row is [x1, y1, x2, y2, confidence, class_id].
    Coordinates are in pixel space of the input image (IMGSZ x IMGSZ).
    """
    results = []
    for i in range(n_real):
        dets = output[i].astype(np.float32)  # [300, 6]
        scores = dets[:, 4]
        class_ids = dets[:, 5].astype(np.int64)

        keep = scores > CONFIDENCE_THRESHOLD
        if not keep.any():
            empty = np.zeros((0, 4), dtype=np.float32)
            results.append((empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)))
            continue

        xyxy = dets[keep, :4]
        cids = class_ids[keep]
        cscores = scores[keep]

        # Remap YOLO class IDs to COCO category IDs
        if YOLO_TO_COCO is not None:
            cids = np.array([YOLO_TO_COCO.get(int(c), int(c)) for c in cids], dtype=np.int64)

        # Scale from IMGSZ space to actual patch size
        ph, pw = patch_sizes[i]
        xyxy[:, 0] *= pw / IMGSZ
        xyxy[:, 1] *= ph / IMGSZ
        xyxy[:, 2] *= pw / IMGSZ
        xyxy[:, 3] *= ph / IMGSZ

        results.append((xyxy, cids, cscores))

    return results


def nms_gpu(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """GPU-accelerated NMS via torchvision. Returns indices to keep."""
    if len(xyxy) == 0:
        return np.array([], dtype=np.int64)
    boxes_t = torch.from_numpy(xyxy.astype(np.float32)).cuda()
    scores_t = torch.from_numpy(scores.astype(np.float32)).cuda()
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    return keep.cpu().numpy()


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------


def compute_patch_grid(image_w: int, image_h: int) -> list[tuple[int, int]]:
    """Compute top-left (x, y) positions for overlapping patches."""

    def _positions(length: int) -> list[int]:
        if length <= PATCH_SIZE:
            return [0]
        overlap = min(MIN_OVERLAP, PATCH_SIZE - 1)
        stride = PATCH_SIZE - overlap
        n = max(1, int(np.ceil((length - PATCH_SIZE) / stride)) + 1)
        if n == 1:
            return [0]
        return np.linspace(0, length - PATCH_SIZE, n).astype(int).tolist()

    return [(x, y) for y in _positions(image_h) for x in _positions(image_w)]


def predict_patched(
    session: ort.InferenceSession, image: np.ndarray, timers: dict
) -> list[dict]:
    """Run batched patch-based inference on a full-resolution image."""
    img_h, img_w = image.shape[:2]

    t0 = time.perf_counter()
    patch_positions = compute_patch_grid(img_w, img_h)

    patches = []
    patch_meta = []  # (px, py, cw, ch)

    for px, py in patch_positions:
        cw = min(PATCH_SIZE, img_w - px)
        ch = min(PATCH_SIZE, img_h - py)
        patch = image[py : py + ch, px : px + cw]

        if cw < PATCH_SIZE or ch < PATCH_SIZE:
            padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            padded[:ch, :cw] = patch
            patch = padded

        patches.append(patch)
        patch_meta.append((px, py, cw, ch))

    timers["patching"] += time.perf_counter() - t0

    # Run inference per patch (ONNX exported with batch=1; YOLO is fast enough)
    input_name = session.get_inputs()[0].name
    all_xyxy = []
    all_scores = []
    all_cids = []

    for idx in range(len(patches)):
        patch = patches[idx]
        px, py, cw, ch = patch_meta[idx]

        # Preprocess single patch
        t0 = time.perf_counter()
        inp = preprocess_batch([patch])  # [1, 3, H, W]
        timers["preprocess"] += time.perf_counter() - t0

        patch_h = ch if ch >= PATCH_SIZE else PATCH_SIZE
        patch_w = cw if cw >= PATCH_SIZE else PATCH_SIZE

        # Run ONNX inference
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: inp})
        timers["session_run"] += time.perf_counter() - t0

        # Decode end2end output [1, 300, 6]
        t0 = time.perf_counter()
        batch_results = decode_end2end(outputs[0], 1, [(patch_h, patch_w)])
        local_xyxy, cids, scores = batch_results[0]
        if len(local_xyxy) == 0:
            timers["postprocess"] += time.perf_counter() - t0
            continue

        # Edge filtering — suppress predictions touching interior patch edges
        touches_left = (local_xyxy[:, 0] < EDGE_TOLERANCE) & (px > 0)
        touches_top = (local_xyxy[:, 1] < EDGE_TOLERANCE) & (py > 0)
        touches_right = (local_xyxy[:, 2] > cw - EDGE_TOLERANCE) & (px + cw < img_w)
        touches_bottom = (local_xyxy[:, 3] > ch - EDGE_TOLERANCE) & (py + ch < img_h)
        not_truncated = ~(touches_left | touches_top | touches_right | touches_bottom)

        local_xyxy = local_xyxy[not_truncated]
        cids = cids[not_truncated]
        scores = scores[not_truncated]
        if len(local_xyxy) == 0:
            timers["postprocess"] += time.perf_counter() - t0
            continue

        # Offset to full-image coordinates and clip
        full_xyxy = local_xyxy.copy()
        full_xyxy[:, 0] = np.clip(full_xyxy[:, 0] + px, 0, img_w)
        full_xyxy[:, 1] = np.clip(full_xyxy[:, 1] + py, 0, img_h)
        full_xyxy[:, 2] = np.clip(full_xyxy[:, 2] + px, 0, img_w)
        full_xyxy[:, 3] = np.clip(full_xyxy[:, 3] + py, 0, img_h)

        # Drop degenerate boxes
        valid = ((full_xyxy[:, 2] - full_xyxy[:, 0]) > 0) & \
                ((full_xyxy[:, 3] - full_xyxy[:, 1]) > 0)

        if valid.any():
            all_xyxy.append(full_xyxy[valid])
            all_scores.append(scores[valid])
            all_cids.append(cids[valid])

        timers["postprocess"] += time.perf_counter() - t0

    if not all_xyxy:
        return []

    # Cross-patch NMS
    t0 = time.perf_counter()
    merged_xyxy = np.concatenate(all_xyxy)
    merged_scores = np.concatenate(all_scores)
    merged_cids = np.concatenate(all_cids)

    keep_idx = nms_gpu(merged_xyxy, merged_scores, STITCH_NMS_IOU)
    merged_xyxy = merged_xyxy[keep_idx]
    merged_scores = merged_scores[keep_idx]
    merged_cids = merged_cids[keep_idx]
    timers["nms"] += time.perf_counter() - t0

    # Build output
    x1 = merged_xyxy[:, 0]
    y1 = merged_xyxy[:, 1]
    bw = merged_xyxy[:, 2] - x1
    bh = merged_xyxy[:, 3] - y1

    preds = [{
        "bbox": [float(x1[j]), float(y1[j]), float(bw[j]), float(bh[j])],
        "category_id": int(merged_cids[j]),
        "confidence": float(merged_scores[j]),
    } for j in range(len(merged_xyxy))]

    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.input)
    output_path = Path(args.output)

    # Load ONNX model
    model_path = Path(__file__).parent / "model.onnx"
    t0 = time.perf_counter()
    session = load_model(str(model_path))
    model_load_time = time.perf_counter() - t0
    print(f"Model loaded in {model_load_time:.1f}s")
    print(f"Config: imgsz={IMGSZ}, half={USE_HALF}, patch={PATCH_SIZE}, overlap={MIN_OVERLAP}")
    if YOLO_TO_COCO:
        print(f"Class mapping: {len(YOLO_TO_COCO)} YOLO→COCO entries")

    # Find all images
    image_paths = sorted(data_dir.glob("*"))
    n_images = len(image_paths)
    print(f"Found {n_images} images")

    timers: dict = defaultdict(float)
    all_predictions = []
    wall_start = time.perf_counter()

    for i, img_path in enumerate(image_paths):
        image_id = int(img_path.stem.split("_")[-1])

        t0 = time.perf_counter()
        image = cv2.imread(str(img_path))
        timers["io"] += time.perf_counter() - t0

        # Downscale if needed
        inv_scale = 1.0
        if MAX_IMAGE_DIM > 0:
            h, w = image.shape[:2]
            max_side = max(h, w)
            if max_side > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max_side
                inv_scale = 1.0 / scale
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        preds = predict_patched(session, image, timers)

        for pred in preds:
            bbox = pred["bbox"]
            if inv_scale != 1.0:
                bbox = [c * inv_scale for c in bbox]
            all_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": pred["category_id"],
                    "bbox": bbox,
                    "score": pred["confidence"],
                }
            )

        if (i + 1) % 10 == 0 or i == n_images - 1:
            elapsed = time.perf_counter() - wall_start
            print(
                f"  [{i + 1}/{n_images}] {elapsed:.0f}s elapsed, {elapsed / (i + 1):.1f}s/img"
            )

    wall_time = time.perf_counter() - wall_start

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(all_predictions, f)

    # Profiling summary
    print(f"\nWrote {len(all_predictions)} predictions to {output_path}")
    print(f"\n--- Profiling ({n_images} images) ---")
    print(f"  Model load:      {model_load_time:6.1f}s")
    for name in ["io", "patching", "preprocess", "session_run", "postprocess", "nms"]:
        print(f"  {name:16s} {timers[name]:6.1f}s")
    accounted = sum(timers.values())
    print(f"  {'other':16s} {wall_time - accounted:6.1f}s")
    print(
        f"  {'TOTAL wall':16s} {wall_time:6.1f}s  ({wall_time / max(n_images, 1):.2f}s/img)"
    )


if __name__ == "__main__":
    main()
