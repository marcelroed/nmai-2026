"""Convert a patched COCO dataset to YOLO txt format for Ultralytics training."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml


def build_class_mapping(
    categories: list[dict],
) -> tuple[dict[int, int], dict[int, int], list[str]]:
    """Build contiguous 0-based class ID mapping from COCO categories.

    COCO category IDs may have gaps (e.g. after label merging). YOLO requires
    contiguous 0-based indices.

    Returns:
        (coco_to_yolo, yolo_to_coco, yolo_names) where:
        - coco_to_yolo: {coco_cat_id: yolo_idx}
        - yolo_to_coco: {yolo_idx: coco_cat_id}
        - yolo_names: list of category names indexed by yolo_idx
    """
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    coco_to_yolo = {}
    yolo_to_coco = {}
    yolo_names = []

    for yolo_idx, cat in enumerate(sorted_cats):
        coco_id = cat["id"]
        coco_to_yolo[coco_id] = yolo_idx
        yolo_to_coco[yolo_idx] = coco_id
        yolo_names.append(cat["name"])

    return coco_to_yolo, yolo_to_coco, yolo_names


def coco_to_yolo_dataset(
    patched_coco_dir: Path,
    output_dir: Path,
) -> tuple[Path, dict[int, int]]:
    """Convert a patched COCO dataset to YOLO format.

    Reads the patched COCO JSON, creates symlinks to patch images, writes
    YOLO label txt files, and generates data.yaml.

    Args:
        patched_coco_dir: Directory containing train/_annotations.coco.json
            (output of prepare_split_datasets with crop_mode="fixed").
        output_dir: Where to write the YOLO dataset.

    Returns:
        (path_to_data_yaml, yolo_to_coco_map)
    """
    # Read patched COCO annotations
    train_json = patched_coco_dir / "train" / "_annotations.coco.json"
    with open(train_json) as f:
        train_coco = json.load(f)

    # Build class mapping from all category IDs that actually appear in
    # annotations. The categories list may be incomplete when label merging
    # removes categories but _replace_annotations brings in annotations
    # that still reference the merged-away IDs.
    cat_name_lookup = {c["id"]: c["name"] for c in train_coco["categories"]}
    ann_cat_ids = {ann["category_id"] for ann in train_coco["annotations"]}
    all_cat_ids = set(cat_name_lookup.keys()) | ann_cat_ids

    # Build a synthetic categories list covering all IDs
    full_categories = []
    for cat_id in sorted(all_cat_ids):
        name = cat_name_lookup.get(cat_id, f"class_{cat_id}")
        full_categories.append({"id": cat_id, "name": name})

    coco_to_yolo, yolo_to_coco, yolo_names = build_class_mapping(full_categories)

    # Create output directories
    train_images_dir = output_dir / "images" / "train"
    train_labels_dir = output_dir / "labels" / "train"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    # Index annotations by image_id
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in train_coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Process each patch image
    for img_info in train_coco["images"]:
        img_id = img_info["id"]
        src_path = Path(img_info["file_name"])
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Symlink image
        dst_path = train_images_dir / src_path.name
        if not dst_path.exists():
            dst_path.symlink_to(src_path)

        # Write YOLO label file
        label_path = train_labels_dir / (src_path.stem + ".txt")
        anns = anns_by_image.get(img_id, [])

        with open(label_path, "w") as f:
            for ann in anns:
                x, y, w, h = ann["bbox"]  # COCO absolute xywh
                coco_cat_id = ann["category_id"]
                yolo_cls = coco_to_yolo[coco_cat_id]

                # Convert to YOLO normalized cxcywh
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                f.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Create a minimal val set (Ultralytics requires it, but real eval
    # uses our competition pipeline via callback)
    val_json = patched_coco_dir / "valid" / "_annotations.coco.json"
    if val_json.exists():
        val_images_dir = output_dir / "images" / "val"
        val_labels_dir = output_dir / "labels" / "val"
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        with open(val_json) as f:
            val_coco = json.load(f)

        val_anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in val_coco["annotations"]:
            val_anns_by_image[ann["image_id"]].append(ann)

        for img_info in val_coco["images"]:
            img_id = img_info["id"]
            src_path = Path(img_info["file_name"])
            img_w = img_info["width"]
            img_h = img_info["height"]

            dst_path = val_images_dir / src_path.name
            if not dst_path.exists():
                dst_path.symlink_to(src_path)

            label_path = val_labels_dir / (src_path.stem + ".txt")
            anns = val_anns_by_image.get(img_id, [])

            with open(label_path, "w") as f:
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    yolo_cls = coco_to_yolo[ann["category_id"]]
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    f.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Write data.yaml
    names_dict = {i: name for i, name in enumerate(yolo_names)}
    data_yaml = output_dir / "data.yaml"
    data_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(yolo_names),
        "names": names_dict,
    }
    with open(data_yaml, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    # Save class mapping for inference
    mapping_file = output_dir / "class_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(
            {
                "coco_to_yolo": {str(k): v for k, v in coco_to_yolo.items()},
                "yolo_to_coco": {str(k): v for k, v in yolo_to_coco.items()},
            },
            f,
            indent=2,
        )

    n_images = len(train_coco["images"])
    n_anns = len(train_coco["annotations"])
    print(
        f"YOLO dataset: {n_images} images, {n_anns} annotations, "
        f"{len(yolo_names)} classes → {output_dir}"
    )

    return data_yaml, yolo_to_coco
