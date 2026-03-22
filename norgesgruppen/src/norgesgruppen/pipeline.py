"""Centralized training and inference functions.

All training and prediction logic lives here so that iterate.py,
train_rfdetr.py, validate.py, etc. are thin consumers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch.multiprocessing

# Seems necessary to enable num_workers > 0
torch.multiprocessing.set_sharing_strategy("file_system")

import numpy as np
import supervision as sv
import torch
from PIL import Image

from norgesgruppen import memeff_hungarian_matcher  # noqa: F401
from norgesgruppen.augmentation import AUG_CONFIG
from norgesgruppen.config import MIN_OVERLAP, MODEL_CLS, PATCH_SIZE
from norgesgruppen.patching import generate_patched_coco, predict_with_patches
from norgesgruppen.postprocess import DEFAULT_TRANSFORMS, Transform, apply_transforms
from norgesgruppen.split import split_coco, write_val_split

# --- Defaults ---
USE_PATCHES = True
PREDICT_THRESHOLD = 0.2
RAW_DATA_DIR = Path("data/train")
PATCHED_DATASET_DIR = Path("data/dataset-patched")
VAL_RATIO = 0.1
SPLIT_SEED = 42


def _patch_cache_key(val_ratio: float, seed: int) -> dict:
    """Build a cache key from the current split config and patch config."""
    return {
        "patch_size": PATCH_SIZE,
        "min_overlap": MIN_OVERLAP,
        "val_ratio": val_ratio,
        "seed": seed,
    }


def _patch_cache_valid(patched_dir: Path, val_ratio: float, seed: int) -> bool:
    """Check if cached patches match the current config."""
    cache_file = patched_dir / ".patch_cache.json"
    if not cache_file.exists():
        return False
    coco_file = patched_dir / "train" / "_annotations.coco.json"
    if not coco_file.exists():
        return False
    val_file = patched_dir / "valid" / "_annotations.coco.json"
    if not val_file.exists():
        return False
    try:
        with open(cache_file) as f:
            cached = json.load(f)
        return cached == _patch_cache_key(val_ratio, seed)
    except (json.JSONDecodeError, KeyError):
        return False


def _write_patch_cache(patched_dir: Path, val_ratio: float, seed: int):
    """Write the cache key after successful patch generation."""
    cache_file = patched_dir / ".patch_cache.json"
    with open(cache_file, "w") as f:
        json.dump(_patch_cache_key(val_ratio, seed), f)


def prepare_data(
    raw_data_dir: Path = RAW_DATA_DIR,
    patched_dir: Path = PATCHED_DATASET_DIR,
    use_patches: bool = USE_PATCHES,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[dict, Path]:
    """Load raw COCO data, split into train/val, and optionally generate patches.

    The split happens BEFORE patching to prevent data leakage.
    Patches are cached — regeneration is skipped if config hasn't changed.

    Returns:
        (coco_dict, dataset_dir) — the full COCO dict (for category info etc.)
        and the dataset directory ready for training.
    """
    # Load raw COCO annotations (always needed for category info)
    annotations_path = Path("data") / "dataset" / "train" / "_annotations.coco.json"
    image_root = raw_data_dir / "images"

    with open(annotations_path) as f:
        coco = json.load(f)

    if _patch_cache_valid(patched_dir, val_ratio, seed):
        print("Dataset cached and up-to-date, skipping generation")
        return coco, patched_dir

    print(
        f"Loaded: {len(coco['images'])} images, {len(coco['annotations'])} annotations"
    )

    # Split into train/val at image level
    train_coco, val_coco = split_coco(
        coco, image_root=image_root, val_ratio=val_ratio, seed=seed
    )

    if use_patches:
        # Patch only the train split
        generate_patched_coco(
            train_coco,
            output_dir=patched_dir,
            patch_size=PATCH_SIZE,
            min_overlap=MIN_OVERLAP,
        )
    else:
        # Write train split without patching
        train_dir = patched_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        with open(train_dir / "_annotations.coco.json", "w") as f:
            json.dump(train_coco, f)

    # Write real val split (no patching — images are symlinked)
    if val_coco["images"]:
        write_val_split(val_coco, output_dir=patched_dir)

    _write_patch_cache(patched_dir, val_ratio, seed)
    return coco, patched_dir


def train(
    dataset_dir: Path,
    output_dir: str = "output",
    epochs: int = 20,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
):
    """Create and train a model. Returns the trained model."""
    model = MODEL_CLS()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        device="cuda",
        run_test=False,
        compile=True,
        early_stopping=False,
        output_dir=output_dir,
        progress_bar=True,
        aug_config=AUG_CONFIG,
        num_workers=8,
    )
    return model


def predict_image(
    model,
    image: Image.Image,
    threshold: float = PREDICT_THRESHOLD,
    use_patches: bool = USE_PATCHES,
    transforms: list[Transform] = DEFAULT_TRANSFORMS,
) -> sv.Detections:
    """Run inference on a single PIL image with optional patching and post-processing."""
    if use_patches:
        detections = predict_with_patches(
            image, lambda img: model.predict(img, threshold=threshold)
        )
    else:
        detections = model.predict(image, threshold=threshold)
    return apply_transforms(detections, transforms)
