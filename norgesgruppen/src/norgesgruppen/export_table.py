"""Fetch the latest 3LC table revision and export to COCO format for training."""

import json
from pathlib import Path

import tlc

PROJECT_NAME = "Norgesgruppen"
DATASET_NAME = "Norgesgruppen Products"
TABLE_NAME = "train"


def export_latest_table(
    output_dir: Path = Path("data/dataset"),
    *,
    dummy_valid: bool = True,
) -> dict:
    """Fetch the latest 3LC table revision and export it as COCO JSON.

    Args:
        output_dir: Directory to write train/ and valid/ annotation files into.
        dummy_valid: If True, create a minimal valid split (1 image) for
            frameworks like RF-DETR that require it but where we don't
            want to waste data on a real val set.

    Returns:
        Tuple of (COCO dict for the training split, table revision name).
    """
    # Fetch latest table revision
    table = tlc.Table.from_names(
        table_name=TABLE_NAME,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
    ).latest()
    print(f"Loaded 3LC table: {len(table)} rows (revision {table.name})")

    # Ensure output dirs exist
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Export full table to COCO
    train_path = train_dir / "_annotations.coco.json"
    table.export(output_url=str(train_path), format="coco", absolute_image_paths=True, include_segmentation=False)

    with open(train_path) as f:
        coco = json.load(f)

    print(f"Exported: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    # Create valid split
    if dummy_valid:
        first_image = coco["images"][0]
        valid_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": [first_image],
            "annotations": [
                a for a in coco["annotations"] if a["image_id"] == first_image["id"]
            ],
        }
        with open(valid_dir / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)

    return coco, table.name

def export_latest_table_mock(
    output_dir: Path = Path("data/dataset"),
    *,
    dummy_valid: bool = True,
) -> dict:
    # Ensure output dirs exist
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"

    train_path = train_dir / "_annotations.coco.json"

    with open(train_path) as f:
        coco = json.load(f)

    print(f"Exported: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    # Create valid split
    if dummy_valid:
        first_image = coco["images"][0]
        valid_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": [first_image],
            "annotations": [
                a for a in coco["annotations"] if a["image_id"] == first_image["id"]
            ],
        }
        with open(valid_dir / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)

    return coco, "LOCAL STATIC RESOURCE"
