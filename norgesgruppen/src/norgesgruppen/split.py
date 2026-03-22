"""Split a COCO dataset into train and validation sets at the image level.

The split happens BEFORE patching to prevent data leakage — no val image
(or any patch of a val image) should ever appear in the training set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def split_coco(
    coco: dict,
    image_root: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Split a COCO dict into train and val at the image level.

    Image file_name fields are resolved to absolute paths using image_root.

    Args:
        coco: Full COCO dict (images, annotations, categories).
        image_root: Directory containing the images (used to resolve relative paths).
        val_ratio: Fraction of images to hold out for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_coco, val_coco) with disjoint image sets and absolute paths.
    """
    images = list(coco["images"])
    rng = random.Random(seed)

    # Shuffle and split
    indices = list(range(len(images)))
    rng.shuffle(indices)

    n_val = round(len(images) * val_ratio)
    val_indices = set(indices[:n_val])

    # Resolve file_name to absolute paths
    def _resolve(img: dict) -> dict:
        img = dict(img)
        p = Path(img["file_name"])
        if not p.is_absolute():
            # file_name may be relative to CWD (e.g. "data/train/images/foo.jpg")
            # or just a basename. Try CWD first, then image_root.
            if (Path.cwd() / p).exists():
                p = (Path.cwd() / p).resolve()
            else:
                p = (image_root / p).resolve()
        img["file_name"] = str(p)
        return img

    val_images = [_resolve(img) for i, img in enumerate(images) if i in val_indices]
    train_images = [_resolve(img) for i, img in enumerate(images) if i not in val_indices]

    val_image_ids = {img["id"] for img in val_images}
    train_image_ids = {img["id"] for img in train_images}

    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_image_ids]
    val_anns = [a for a in coco["annotations"] if a["image_id"] in val_image_ids]

    base = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
    }

    train_coco = {**base, "images": train_images, "annotations": train_anns}
    val_coco = {**base, "images": val_images, "annotations": val_anns}

    print(
        f"Split: {len(train_images)} train images ({len(train_anns)} anns), "
        f"{len(val_images)} val images ({len(val_anns)} anns)"
    )

    return train_coco, val_coco


def write_val_split(val_coco: dict, output_dir: Path) -> None:
    """Write the validation COCO split to output_dir/valid/.

    Symlinks each val image into the valid/ directory so that dataset
    loaders expecting images alongside annotations can find them.
    """
    valid_dir = output_dir / "valid"
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Symlink images and update file_name to point to the symlink
    updated_images = []
    for img in val_coco["images"]:
        src = Path(img["file_name"])
        dst = valid_dir / src.name
        # Remove stale symlink if it exists
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)

        updated = dict(img)
        updated["file_name"] = str(dst.resolve())
        updated_images.append(updated)

    out_coco = {
        "info": val_coco.get("info", {}),
        "licenses": val_coco.get("licenses", []),
        "categories": val_coco["categories"],
        "images": updated_images,
        "annotations": val_coco["annotations"],
    }

    with open(valid_dir / "_annotations.coco.json", "w") as f:
        json.dump(out_coco, f)

    print(f"Val split written: {len(updated_images)} images in {valid_dir}")
