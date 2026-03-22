"""One iteration of the data-driven loop: export table → train → collect metrics."""

import gc

import tlc
import torch
import typer
from PIL import Image

from norgesgruppen.pipeline import predict_image, prepare_data, train

# --- Config ---
OUTPUT_DIR = "output-iterate"
EPOCHS = 2
BATCH_SIZE = 8
LR = 1e-4


def collect_metrics(model, table: tlc.Table, classes: list[str]):
    """Run inference on every sample and write per-sample metrics to a 3LC run."""
    run = tlc.init(project_name="Norgesgruppen")

    model.optimize_for_inference()

    predicted_bbs = []

    with torch.no_grad():
        for i in range(len(table)):
            sample = table[i]
            image = Image.open(sample["image"])
            detections = predict_image(model, image)

            bb_list = []
            if not detections.is_empty():
                for j in range(len(detections)):
                    x0, y0, x1, y1 = detections.xyxy[j]
                    bb_list.append({
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                        "label": int(detections.class_id[j]),
                        "confidence": float(detections.confidence[j]),
                    })

            predicted_bbs.append({
                "bb_list": bb_list,
                "image_width": sample["width"],
                "image_height": sample["height"],
            })

            if (i + 1) % 10 == 0:
                print(f"  Inference: {i + 1}/{len(table)}")

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


def main(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
):
    coco, dataset_dir = prepare_data()
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    classes = [c["name"] for c in cats]
    print(f"Classes: {len(classes)}")

    model = train(dataset_dir, output_dir=OUTPUT_DIR, epochs=epochs,
                  batch_size=batch_size, lr=lr)

    gc.collect()
    torch.cuda.empty_cache()

    table = tlc.Table.from_names(
        table_name="train",
        dataset_name="Norgesgruppen Products",
        project_name="Norgesgruppen",
    ).latest()

    collect_metrics(model, table, classes)
    print("Done! Check the 3LC dashboard for per-sample metrics.")


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
