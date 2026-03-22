"""Collect 3LC metrics from run.py prediction output (no model loading required)."""

import json
import re
from collections import defaultdict
from pathlib import Path

import tlc
import typer

INVALID_CHARS = re.compile(r'[<>\\|.:"\'?*&]')


def main(
    predictions: Path = typer.Argument(..., help="Path to predictions JSON from run.py"),
):
    # Load predictions and group by image_id
    with open(predictions) as f:
        preds_list = json.load(f)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for p in preds_list:
        preds_by_image[p["image_id"]].append(p)

    # Load COCO annotations for category names
    annotations_path = Path("data/train/annotations.json")
    with open(annotations_path) as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    classes = [INVALID_CHARS.sub("", c["name"]) for c in cats]

    # Build image_id -> (width, height) lookup
    image_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    # Load the 3LC table
    table = tlc.Table.from_names(
        table_name="train",
        dataset_name="Norgesgruppen Products",
        project_name="Norgesgruppen",
    ).latest()

    run = tlc.init(project_name="Norgesgruppen")

    predicted_bbs = []
    for i in range(len(table)):
        sample = table[i]
        image_path = Path(sample["image"])
        # Extract image_id from filename (e.g. "img_00001.jpg" -> 1)
        image_id = int(image_path.stem.split("_")[-1])

        width = sample["width"]
        height = sample["height"]

        bb_list = []
        for p in preds_by_image.get(image_id, []):
            x, y, w, h = p["bbox"]
            bb_list.append({
                "x0": float(x),
                "y0": float(y),
                "x1": float(x + w),
                "y1": float(y + h),
                "label": int(p["category_id"]),
                "confidence": float(p["score"]),
            })

        predicted_bbs.append({
            "bb_list": bb_list,
            "image_width": width,
            "image_height": height,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed: {i + 1}/{len(table)}")

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


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
