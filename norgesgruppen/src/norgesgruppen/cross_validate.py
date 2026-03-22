"""2-fold cross-validation: train on each half, predict on the other, collect 3LC metrics.

After each fold's training the best EMA checkpoint is exported to ONNX (fp16)
and inference is run using the same patched ONNX pipeline as submission/run.py,
so that the predictions match the submission flow exactly.
"""

import argparse
import gc
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import onnx
import tlc
import torch
import typer
from onnxruntime.transformers.float16 import convert_float_to_float16

from norgesgruppen.config import EXCLUDED_IMAGE_IDS, MIN_OVERLAP, MODEL_CLS, NUM_CLASSES, PATCH_SIZE
from norgesgruppen.patching import generate_patched_coco
from norgesgruppen.pipeline import train
from norgesgruppen.split import split_coco, write_val_split

RAW_DATA_DIR = Path("data/train")
SEED = 42
EPOCHS = 9
BATCH_SIZE = 4
LR = 1e-4
ONNX_BATCH_SIZE = 16
INVALID_CHARS = re.compile(r'[<>\\|.:"\'?*&]')


def _export_onnx_fp16(checkpoint: Path, onnx_dir: Path) -> Path:
    """Export a training checkpoint to an fp16 ONNX model (mirrors prepare_submission.py)."""
    model = MODEL_CLS()
    model.model.reinitialize_detection_head(num_classes=NUM_CLASSES)

    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.model.model.load_state_dict(ckpt[state_key])
    print(f"Loaded {state_key} weights (epoch {ckpt.get('epoch', '?')})")

    onnx_dir.mkdir(parents=True, exist_ok=True)
    model.export(output_dir=str(onnx_dir), batch_size=ONNX_BATCH_SIZE)

    onnx_path = next(onnx_dir.glob("*.onnx"))
    print(f"FP32 ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    onnx_model = onnx.load(str(onnx_path))
    onnx_model_fp16 = convert_float_to_float16(onnx_model, keep_io_types=False)
    fp16_path = onnx_dir / "model_fp16.onnx"
    onnx.save(onnx_model_fp16, str(fp16_path))
    print(f"FP16 ONNX: {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")

    # Free model from GPU
    del model, ckpt, onnx_model, onnx_model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    return fp16_path


def _run_onnx_inference(model_onnx: Path, image_dir: Path, output_json: Path):
    """Run submission/run.py on a directory of images."""
    run_py = Path("submission/run.py")
    # Use a temporary symlink/copy of the model so run.py finds it next to itself
    model_link = run_py.parent / "model.onnx"
    already_existed = model_link.exists()
    old_target = model_link.resolve() if already_existed else None

    model_link.unlink(missing_ok=True)
    model_link.symlink_to(model_onnx.resolve())

    try:
        subprocess.run(
            [
                sys.executable, str(run_py),
                "--input", str(image_dir),
                "--output", str(output_json),
            ],
            check=True,
        )
    finally:
        # Restore previous model.onnx if there was one
        model_link.unlink(missing_ok=True)
        if already_existed and old_target:
            model_link.symlink_to(old_target)


def _load_run_predictions(predictions_json: Path) -> dict[int, list[dict]]:
    """Load run.py output and convert to 3LC bounding box format, keyed by image_id."""
    with open(predictions_json) as f:
        preds_list = json.load(f)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for p in preds_list:
        x, y, w, h = p["bbox"]
        preds_by_image[p["image_id"]].append({
            "x0": float(x),
            "y0": float(y),
            "x1": float(x + w),
            "y1": float(y + h),
            "label": int(p["category_id"]),
            "confidence": float(p["score"]),
        })
    return dict(preds_by_image)


def _filter_excluded_images(coco: dict, exclude_ids: set[str]) -> dict:
    """Remove images (and their annotations) by image ID."""
    if not exclude_ids:
        return coco

    kept_images = [img for img in coco["images"] if img["file_name"] not in exclude_ids]
    kept_image_ids = {img["id"] for img in kept_images}
    kept_anns = [a for a in coco["annotations"] if a["image_id"] in kept_image_ids]

    removed = len(coco["images"]) - len(kept_images)
    print(f"Excluded {removed} images ({len(exclude_ids)} IDs requested, {removed} matched)")

    return {
        **coco,
        "images": kept_images,
        "annotations": kept_anns,
    }


def main(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = SEED,
    exclude_ids_file: Path | None = None,
):
    # Load COCO annotations
    with open(RAW_DATA_DIR / "annotations.json") as f:
        coco = json.load(f)
    image_root = RAW_DATA_DIR / "images"

    # Merge exclusions: central config + optional extra file
    exclude_ids = set(EXCLUDED_IMAGE_IDS)
    if exclude_ids_file is not None:
        raw = exclude_ids_file.read_text().strip().splitlines()
        exclude_ids |= {line.strip() for line in raw if line.strip()}

    coco = _filter_excluded_images(coco, exclude_ids)

    # 50/50 split
    half_a, half_b = split_coco(coco, image_root, val_ratio=0.5, seed=seed)
    folds = [(half_a, half_b), (half_b, half_a)]

    all_preds: dict[int, list[dict]] = {}

    for fold_idx, (train_coco, val_coco) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}: train={len(train_coco['images'])} images, val={len(val_coco['images'])} images")
        print(f"{'='*60}\n")

        fold_dir = Path(f"data/dataset-cv-fold{fold_idx}")
        output_dir = f"output-cv-fold{fold_idx}"

        # Prepare dataset directory for this fold
        generate_patched_coco(
            train_coco,
            output_dir=fold_dir,
            patch_size=PATCH_SIZE,
            min_overlap=MIN_OVERLAP,
        )
        write_val_split(val_coco, output_dir=fold_dir)

        # Train
        model = train(
            fold_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Export best checkpoint to ONNX fp16
        checkpoint = Path(output_dir) / "checkpoint_best_ema.pth"
        if not checkpoint.exists():
            checkpoint = Path(output_dir) / "checkpoint.pth"
        onnx_dir = Path(output_dir) / "onnx"
        onnx_path = _export_onnx_fp16(checkpoint, onnx_dir)

        # Prepare a temporary directory with just the val images for run.py
        val_image_dir = fold_dir / "val_images"
        val_image_dir.mkdir(parents=True, exist_ok=True)
        for img_info in val_coco["images"]:
            src = Path(img_info["file_name"])
            dst = val_image_dir / src.name
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)

        # Run ONNX inference (same as run.py)
        preds_json = Path(output_dir) / "val_predictions.json"
        _run_onnx_inference(onnx_path, val_image_dir, preds_json)

        # Collect predictions
        fold_preds = _load_run_predictions(preds_json)
        all_preds.update(fold_preds)

        # Mark images with no predictions
        for img_info in val_coco["images"]:
            if img_info["id"] not in all_preds:
                all_preds[img_info["id"]] = []

        n_boxes = sum(len(v) for v in fold_preds.values())
        print(f"Fold {fold_idx}: {n_boxes} predictions on {len(val_coco['images'])} val images")

    # Collect all predictions as 3LC metrics
    print(f"\nCollecting 3LC metrics for {len(all_preds)} images...")

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    classes = [INVALID_CHARS.sub("", c["name"]) for c in cats]

    table = tlc.Table.from_names(
        table_name="train",
        dataset_name="Norgesgruppen Products",
        project_name="Norgesgruppen",
    ).latest()

    run = tlc.init(project_name="Norgesgruppen")

    predicted_bbs = []
    for i in range(len(table)):
        sample = table[i]
        image_id = int(Path(sample["image"]).stem.split("_")[-1])

        predicted_bbs.append({
            "bb_list": all_preds.get(image_id, []),
            "image_width": sample["width"],
            "image_height": sample["height"],
        })

    run.add_metrics(
        metrics={"predicted_bbs": predicted_bbs},
        column_schemas={
            "predicted_bbs": tlc.BoundingBoxListSchema(
                classes=classes,
                is_prediction=True,
                include_segmentation=False,
            ),
        },
        foreign_table_url=table.url,
    )

    print(f"Metrics written to run: {run.url}")
    print("Done! All images have out-of-fold predictions.")


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
