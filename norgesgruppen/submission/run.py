# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch==2.6.0+cu124",
#     "torchvision==0.21.0+cu124",
#     "ultralytics==8.1.0",
#     "onnxruntime-gpu==1.20.0",
#     "opencv-python-headless==4.9.0.80",
#     "albumentations==1.3.1",
#     "pillow==10.2.0",
#     "numpy==1.26.4",
#     "scipy==1.12.0",
#     "scikit-learn==1.4.0",
#     "pycocotools==2.0.7",
#     "ensemble-boxes==1.0.9",
#     "timm==0.9.12",
#     "supervision==0.18.0",
#     "safetensors==0.4.2",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu124"]
# ///


# Run this with for instance with `uv run --no-project submission/run.py --input data/train/images --output predictions.json`

"""Competition submission entry point.

Runs RF-DETR 2XLarge (ONNX, FP16) with batched patch-based inference on full-resolution images.

Optimizations over baseline:
  - GPU-accelerated NMS (torchvision.ops.nms) instead of O(n²) Python NMS
  - Vectorized edge filtering and decode (numpy, no per-prediction Python loops)
  - Optional horizontal-flip TTA (HFLIP_TTA flag — doubles patches, ~1.8x time)

Contract:
    python run.py --input /data/images --output /output/predictions.json

Sandbox: Python 3.11, onnxruntime-gpu 1.20.0, PyTorch 2.6.0+cu124, NVIDIA L4 24GB,
         no network, 300s timeout.
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

# --- Model config (RF-DETR 2XLarge) ---
RESOLUTION = 880
MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 0.4
NUM_CLASSES = 358  # 357 product categories (0-356) + 1 background (rfdetr convention)
MAX_CATEGORY_ID = 355  # Highest valid product category_id per competition spec (0-355)

# --- Label merge map (variant → canonical) ---
# Set APPLY_LABEL_MERGES = True when using a model trained with --merge-labels
APPLY_LABEL_MERGES = False
LABEL_MERGE_MAP = {
    59: 61,    # MÜSLI BLÅBÆR 630G AXA → MUSLI BLÅBÆR 630G AXA
    170: 260,  # MÜSLI ENERGI 650G AXA → MUSLI ENERGI 675G AXA
    36: 201,   # MÜSLI FRUKT MÜSLI 700G AXA → MUSLI FRUKT 700G AXA
}

# --- Inference config ---
BATCH_SIZE = 16  # Fixed batch size matching the ONNX export

# --- Patching config ---
PATCH_SIZE = RESOLUTION  # Patches match model input resolution
MIN_OVERLAP = 440  # pixels — 50% overlap
EDGE_TOLERANCE = 3  # pixels — predictions touching patch edge are dropped
STITCH_NMS_IOU = 0.6

# --- Image downscaling (must match training --max-image-dim) ---
MAX_IMAGE_DIM = 4000  # Must match training --max-image-dim; 0 = no limit

# --- TTA config ---
# Horizontal flip TTA: ~+0.006 combined score, but ~1.8x inference time.
# On L4 with 248 images: baseline ~200s, hflip ~356s (over 300s budget).
# Enable only if you have timing headroom or fewer test images.
HFLIP_TTA = False


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def load_model(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"ORT providers: {session.get_providers()}")
    return session


def preprocess_batch_gpu(patches: list[np.ndarray]) -> torch.Tensor:
    """Preprocess a list of HWC uint8 patches on GPU. Returns [N, 3, H, W] float16 tensor."""
    resized = []
    for p in patches:
        if p.shape[0] != RESOLUTION or p.shape[1] != RESOLUTION:
            p = cv2.resize(p, (RESOLUTION, RESOLUTION))
        resized.append(p)
    batch = torch.from_numpy(np.stack(resized)).cuda()
    batch = batch.to(torch.float32) / 255.0
    means = torch.tensor(MEANS, device="cuda")
    stds = torch.tensor(STDS, device="cuda")
    batch = (batch - means) / stds
    batch = batch.permute(0, 3, 1, 2)  # NHWC -> NCHW
    batch = batch.to(torch.float16).contiguous()
    return batch


def decode_batch_vectorized(
    pred_boxes: np.ndarray,
    pred_logits: np.ndarray,
    orig_sizes: list[tuple[int, int]],
    n_real: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode predictions for a batch. Returns (local_xyxy, cids, scores) per image.

    Fully vectorized — no per-prediction Python loops.
    """
    pred_logits = pred_logits[:n_real].astype(np.float32)
    pred_boxes = pred_boxes[:n_real].astype(np.float32)

    # Sigmoid on logits
    scores = 1.0 / (1.0 + np.exp(-pred_logits))

    results = []
    for i in range(n_real):
        orig_h, orig_w = orig_sizes[i]
        s = scores[i]  # [Q, C]
        class_ids = s.argmax(axis=1)
        class_scores = s.max(axis=1)

        keep = (class_scores > CONFIDENCE_THRESHOLD) & (class_ids <= MAX_CATEGORY_ID)
        if not keep.any():
            empty = np.zeros((0, 4), dtype=np.float32)
            results.append((empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)))
            continue

        boxes = pred_boxes[i][keep]
        cids = class_ids[keep]
        cscores = class_scores[keep]

        if APPLY_LABEL_MERGES:
            for old_id, new_id in LABEL_MERGE_MAP.items():
                cids[cids == old_id] = new_id

        # cxcywh → xyxy in pixel coords
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        local_xyxy = np.stack([
            (cx - w / 2) * orig_w,
            (cy - h / 2) * orig_h,
            (cx + w / 2) * orig_w,
            (cy + h / 2) * orig_h,
        ], axis=1)

        results.append((local_xyxy, cids, cscores))

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
    session: ort.InferenceSession, image: np.ndarray, timers: dict[str, float]
) -> list[dict]:
    """Run batched patch-based inference with optional hflip TTA."""
    img_h, img_w = image.shape[:2]

    t0 = time.perf_counter()
    patch_positions = compute_patch_grid(img_w, img_h)

    patches = []
    patch_meta = []  # (px, py, cw, ch, is_flipped)

    for px, py in patch_positions:
        cw = min(PATCH_SIZE, img_w - px)
        ch = min(PATCH_SIZE, img_h - py)
        patch = image[py : py + ch, px : px + cw]

        if cw < PATCH_SIZE or ch < PATCH_SIZE:
            padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            padded[:ch, :cw] = patch
            patch = padded

        patches.append(patch)
        patch_meta.append((px, py, cw, ch, False))

        if HFLIP_TTA:
            patches.append(patch[:, ::-1].copy())
            patch_meta.append((px, py, cw, ch, True))

    timers["patching"] += time.perf_counter() - t0

    # Preprocess ALL patches upfront on GPU
    t0 = time.perf_counter()
    n_real = len(patches)
    if n_real % BATCH_SIZE != 0:
        pad_count = BATCH_SIZE - (n_real % BATCH_SIZE)
        dummy = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        patches.extend([dummy] * pad_count)
    all_preprocessed = preprocess_batch_gpu(patches)
    timers["preprocess"] += time.perf_counter() - t0

    # Run inference in batches
    input_name = session.get_inputs()[0].name
    all_xyxy = []
    all_scores = []
    all_cids = []

    for batch_start in range(0, len(patches), BATCH_SIZE):
        n_real_in_batch = min(BATCH_SIZE, n_real - batch_start)
        if n_real_in_batch <= 0:
            break

        batch_tensor = all_preprocessed[batch_start : batch_start + BATCH_SIZE]

        batch_sizes = []
        for i in range(batch_start, batch_start + n_real_in_batch):
            _, _, cw, ch, _ = patch_meta[i]
            decode_h = PATCH_SIZE if ch < PATCH_SIZE else ch
            decode_w = PATCH_SIZE if cw < PATCH_SIZE else cw
            batch_sizes.append((decode_h, decode_w))

        # IO binding: feed GPU tensor directly, avoid CPU roundtrip
        t0 = time.perf_counter()
        io_binding = session.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float16,
            shape=tuple(batch_tensor.shape),
            buffer_ptr=batch_tensor.data_ptr(),
        )
        for out in session.get_outputs():
            io_binding.bind_output(out.name, device_type="cpu")
        session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()
        timers["session_run"] += time.perf_counter() - t0

        # Decode (vectorized)
        t0 = time.perf_counter()
        batch_results = decode_batch_vectorized(
            outputs[0], outputs[1], batch_sizes, n_real_in_batch
        )

        for idx in range(n_real_in_batch):
            local_xyxy, cids, scores = batch_results[idx]
            if len(local_xyxy) == 0:
                continue

            px, py, cw, ch, is_flipped = patch_meta[batch_start + idx]

            # Flip x-coordinates back for flipped patches
            if is_flipped:
                new_x1 = cw - local_xyxy[:, 2]
                new_x2 = cw - local_xyxy[:, 0]
                local_xyxy[:, 0] = new_x1
                local_xyxy[:, 2] = new_x2

            # Vectorized edge filtering — suppress boxes touching interior patch edges
            touches_left   = (local_xyxy[:, 0] < EDGE_TOLERANCE) & (px > 0)
            touches_top    = (local_xyxy[:, 1] < EDGE_TOLERANCE) & (py > 0)
            touches_right  = (local_xyxy[:, 2] > cw - EDGE_TOLERANCE) & (px + cw < img_w)
            touches_bottom = (local_xyxy[:, 3] > ch - EDGE_TOLERANCE) & (py + ch < img_h)
            not_truncated = ~(touches_left | touches_top | touches_right | touches_bottom)

            local_xyxy = local_xyxy[not_truncated]
            cids = cids[not_truncated]
            scores = scores[not_truncated]
            if len(local_xyxy) == 0:
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

    # GPU-accelerated NMS
    t0 = time.perf_counter()
    merged_xyxy = np.concatenate(all_xyxy)
    merged_scores = np.concatenate(all_scores)
    merged_cids = np.concatenate(all_cids)

    keep_idx = nms_gpu(merged_xyxy, merged_scores, STITCH_NMS_IOU)
    merged_xyxy = merged_xyxy[keep_idx]
    merged_scores = merged_scores[keep_idx]
    merged_cids = merged_cids[keep_idx]
    timers["nms"] += time.perf_counter() - t0

    # Build output dicts
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

    # Load ONNX model (same directory as run.py)
    model_path = Path(__file__).parent / "model.onnx"
    t0 = time.perf_counter()
    session = load_model(str(model_path))
    model_load_time = time.perf_counter() - t0
    print(f"Model loaded in {model_load_time:.1f}s")

    # Find all images
    image_paths = sorted(data_dir.glob("*"))
    n_images = len(image_paths)
    print(f"Found {n_images} images (hflip_tta={HFLIP_TTA})")

    timers: dict[str, float] = defaultdict(float)
    all_predictions = []
    wall_start = time.perf_counter()

    for i, img_path in enumerate(image_paths):
        image_id = int(img_path.stem.split("_")[-1])

        t0 = time.perf_counter()
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        timers["io"] += time.perf_counter() - t0

        # Downscale if needed (must match training)
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
