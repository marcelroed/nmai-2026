"""Validation script: evaluate a model using the competition scoring metric.

Two modes:
  1. Full evaluation: run the model on all training data (no overfitting check).
  2. Holdout / k-fold: hold out a small set, train on the rest, evaluate on holdout.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
import typer
from PIL import Image

from norgesgruppen.config import MODEL_CLS, PATCH_SIZE, MIN_OVERLAP
from norgesgruppen.patching import generate_patched_coco
from norgesgruppen.pipeline import predict_image, train
from norgesgruppen.scoring import ScoreResult, compute_score


def sample_to_gt(sample: dict) -> dict:
    """Convert a 3LC table sample to a ground truth dict for scoring."""
    bbs = sample["bbs"]
    if not bbs["bb_list"]:
        return {
            "boxes": np.zeros((0, 4)),
            "labels": np.zeros((0,), dtype=int),
        }

    # 3LC stores boxes as x0, y0, x1, y1 where x1/y1 are actually width/height
    boxes = np.array([[bb["x0"], bb["y0"], bb["x0"] + bb["x1"], bb["y0"] + bb["y1"]] for bb in bbs["bb_list"]])
    labels = np.array([bb["label"] for bb in bbs["bb_list"]], dtype=int)
    return {"boxes": boxes, "labels": labels}


def _predict_for_scoring(model, image_path: str) -> dict:
    """Run model on a single image, return dict for scoring."""
    image = Image.open(image_path)
    detections = predict_image(model, image)

    if detections.is_empty():
        return {
            "boxes": np.zeros((0, 4)),
            "labels": np.zeros((0,), dtype=int),
            "scores": np.zeros((0,)),
        }

    return {
        "boxes": detections.xyxy,
        "labels": detections.class_id,
        "scores": detections.confidence,
    }


def evaluate(model, samples: list[dict]) -> ScoreResult:
    """Evaluate model on a list of 3LC samples."""
    ground_truths = []
    predictions = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            ground_truths.append(sample_to_gt(sample))
            predictions.append(_predict_for_scoring(model, sample["image"]))

            if (i + 1) % 10 == 0:
                print(f"  Evaluating: {i + 1}/{len(samples)}")

    return compute_score(ground_truths, predictions)


def evaluate_full(checkpoint_dir: str = "output-iterate") -> None:
    """Evaluate best checkpoint on all training data."""
    import tlc

    table = tlc.Table.from_names(
        table_name="train",
        dataset_name="Norgesgruppen Products",
        project_name="Norgesgruppen",
    ).latest()

    model = MODEL_CLS()
    checkpoint = Path(checkpoint_dir) / "checkpoint_best_total.pth"
    if checkpoint.exists():
        model.load(str(checkpoint))
        print(f"Loaded checkpoint: {checkpoint}")

    samples = [table[i] for i in range(len(table))]
    result = evaluate(model, samples)
    print(f"\nFull dataset evaluation:\n{result}")


def evaluate_holdout(
    n_holdout: int = 15,
    n_folds: int = 1,
    epochs: int = 10,
    batch_size: int = 2,
    seed: int = 42,
) -> None:
    """Train on subset, evaluate on holdout. Optionally average across folds."""
    import tlc

    table = tlc.Table.from_names(
        table_name="train",
        dataset_name="Norgesgruppen Products",
        project_name="Norgesgruppen",
    ).latest()

    n_total = len(table)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    fold_results = []

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")

        start = (fold * n_holdout) % n_total
        holdout_idx = set(indices[start:start + n_holdout].tolist())
        train_idx = [i for i in range(n_total) if i not in holdout_idx]

        print(f"Train: {len(train_idx)} images, Holdout: {len(holdout_idx)} images")

        coco = _export_subset(table, train_idx)

        # Prepare dataset (with optional patching)
        dataset_dir = Path("data/dataset-validate")
        _write_coco_split(coco, dataset_dir)
        patched_dir = Path("data/dataset-validate-patched")
        generate_patched_coco(coco, output_dir=patched_dir, patch_size=PATCH_SIZE, min_overlap=MIN_OVERLAP)
        train_dir = patched_dir

        model = train(train_dir, output_dir=f"output-validate-fold{fold}",
                      epochs=epochs, batch_size=batch_size,
                      grad_accum_steps=max(1, 16 // batch_size))

        holdout_samples = [table[i] for i in sorted(holdout_idx)]
        result = evaluate(model, holdout_samples)
        fold_results.append(result)
        print(f"\nFold {fold + 1} result:\n{result}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if n_folds > 1:
        avg_det = np.mean([r.detection_map for r in fold_results])
        avg_cls = np.mean([r.classification_map for r in fold_results])
        avg_combined = np.mean([r.combined for r in fold_results])
        print(f"\n{'='*60}")
        print(f"Average across {n_folds} folds:")
        print(f"  detection mAP@0.5: {avg_det:.4f}")
        print(f"  classification mAP@0.5: {avg_cls:.4f}")
        print(f"  combined: {avg_combined:.4f}")


def _export_subset(table, indices: list[int]) -> dict:
    """Export a subset of the table to COCO format in memory."""
    import json

    tmp_path = Path("data/_tmp_full_export.json")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    table.export(output_url=str(tmp_path), format="coco", absolute_image_paths=True, include_segmentation=False)

    with open(tmp_path) as f:
        coco = json.load(f)

    keep_image_ids = set()
    for idx in indices:
        if idx < len(coco["images"]):
            keep_image_ids.add(coco["images"][idx]["id"])

    coco["images"] = [img for img in coco["images"] if img["id"] in keep_image_ids]
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] in keep_image_ids]

    tmp_path.unlink(missing_ok=True)
    return coco


def _write_coco_split(coco: dict, dataset_dir: Path) -> None:
    """Write COCO dict as train split + dummy valid split."""
    import json

    for split in ("train", "valid"):
        (dataset_dir / split).mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "train" / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)

    first_image = coco["images"][0]
    valid_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [first_image],
        "annotations": [a for a in coco["annotations"] if a["image_id"] == first_image["id"]],
    }
    with open(dataset_dir / "valid" / "_annotations.coco.json", "w") as f:
        json.dump(valid_coco, f)


app = typer.Typer()


@app.command()
def full(checkpoint_dir: str = "output-iterate"):
    """Evaluate on full training data (no overfitting check)."""
    evaluate_full(checkpoint_dir)


@app.command()
def holdout(
    n_holdout: int = 15,
    n_folds: int = 1,
    epochs: int = 10,
    batch_size: int = 2,
    seed: int = 42,
):
    """Train/eval with holdout validation."""
    evaluate_holdout(n_holdout, n_folds, epochs, batch_size, seed)


if __name__ == "__main__":
    app()
