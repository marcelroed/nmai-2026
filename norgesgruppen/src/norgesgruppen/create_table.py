import json
import re
import tempfile
from pathlib import Path

import tlc

PROJECT_NAME = "Norgesgruppen"
DATASET_NAME = "Norgesgruppen Products"
TABLE_NAME = "train"

INVALID_CHARS = re.compile(r'[<>\\|.:"\'?*&]')

annotations_file = (Path() / "data" / "train" / "annotations.json").resolve()
image_folder = (Path() / "data" / "train" / "images").resolve()

assert annotations_file.exists(), f"Annotations file not found: {annotations_file}"
assert image_folder.exists(), f"Image folder not found: {image_folder}"

# Sanitize category names to remove characters disallowed by 3LC
with open(annotations_file) as f:
    coco_data = json.load(f)

for category in coco_data["categories"]:
    category["name"] = INVALID_CHARS.sub("", category["name"])

# Add mock segmentation polygons (bbox outline) where missing
for annotation in coco_data["annotations"]:
    if "segmentation" not in annotation or not annotation["segmentation"]:
        x, y, w, h = annotation["bbox"]
        annotation["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]

sanitized_annotations = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False
)
with sanitized_annotations:
    json.dump(coco_data, sanitized_annotations)

table = tlc.Table.from_coco(
    annotations_file=sanitized_annotations.name,
    image_folder=image_folder,
    table_name=TABLE_NAME,
    dataset_name=DATASET_NAME,
    project_name=PROJECT_NAME,
    task="detect",
    # We do not want to accidentally overwrite the existing table
    if_exists="raise",
)

Path(sanitized_annotations.name).unlink()

print(f"Created table with {len(table)} rows")
