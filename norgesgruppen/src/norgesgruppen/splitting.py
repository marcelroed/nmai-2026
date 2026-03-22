"""Stratified image-level train/val split for multi-label object detection.

Splits COCO annotations at the IMAGE level (not patch level) using iterative
stratification to ensure balanced category representation in both sets.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from norgesgruppen.config import MIN_OVERLAP, PATCH_SIZE
from norgesgruppen.patching import generate_patched_coco

# ---------------------------------------------------------------------------
# Images excluded from both training and validation
# ---------------------------------------------------------------------------
# img_00295.jpg (id=295): Dense knekkebrød shelf with 136 annotations, many
# with incorrect bounding boxes and category labels beyond practical repair.
EXCLUDED_IMAGE_IDS: set[int] = {295}


def _exclude_images(coco: dict) -> dict:
    """Remove excluded images and their annotations from a COCO dict."""
    if not EXCLUDED_IMAGE_IDS:
        return coco
    kept_images = [img for img in coco["images"] if img["id"] not in EXCLUDED_IMAGE_IDS]
    kept_anns = [a for a in coco["annotations"] if a["image_id"] not in EXCLUDED_IMAGE_IDS]
    n_removed = len(coco["images"]) - len(kept_images)
    if n_removed:
        print(f"Excluded {n_removed} image(s) (ids {EXCLUDED_IMAGE_IDS}): "
              f"{len(coco['images'])} → {len(kept_images)} images, "
              f"{len(coco['annotations'])} → {len(kept_anns)} annotations")
    return {**coco, "images": kept_images, "annotations": kept_anns}


# ---------------------------------------------------------------------------
# Label merging — collapse duplicate/variant categories into canonical ones
# ---------------------------------------------------------------------------
LABEL_MERGE_MAP: dict[int, int] = {
    59: 61,    # MÜSLI BLÅBÆR 630G AXA → MUSLI BLÅBÆR 630G AXA
    170: 260,  # MÜSLI ENERGI 650G AXA → MUSLI ENERGI 675G AXA
    36: 201,   # MÜSLI FRUKT MÜSLI 700G AXA → MUSLI FRUKT 700G AXA
}


def apply_label_merges(coco: dict) -> dict:
    """Remap annotations with variant category IDs to their canonical IDs.

    Also removes merged-away categories from the categories list so the
    dataset has fewer unique categories (model head size stays unchanged).
    """
    if not LABEL_MERGE_MAP:
        return coco

    merged_away = set(LABEL_MERGE_MAP.keys())
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Remap annotations
    new_anns = []
    n_remapped = 0
    for ann in coco["annotations"]:
        if ann["category_id"] in LABEL_MERGE_MAP:
            ann = {**ann, "category_id": LABEL_MERGE_MAP[ann["category_id"]]}
            n_remapped += 1
        new_anns.append(ann)

    # Remove merged-away categories
    new_cats = [c for c in coco["categories"] if c["id"] not in merged_away]

    print(f"Label merges: remapped {n_remapped} annotations, "
          f"removed {len(merged_away)} categories ({len(coco['categories'])} → {len(new_cats)})")
    for old_id, new_id in LABEL_MERGE_MAP.items():
        print(f"  {cat_names.get(old_id, old_id)} (id={old_id}) → "
              f"{cat_names.get(new_id, new_id)} (id={new_id})")

    return {**coco, "categories": new_cats, "annotations": new_anns}


def iterative_stratification(
    coco: dict,
    val_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Split a COCO dict into train/val at image level using iterative stratification.

    Categories appearing in only one image are forced into train (can't validate
    on a category never seen in training). Remaining images are distributed to
    balance category representation across splits.

    Returns:
        (train_coco, val_coco) with disjoint image sets, shared categories list.
    """
    rng = np.random.default_rng(seed)

    # Build image_id -> index mapping
    images = coco["images"]
    image_ids = [img["id"] for img in images]
    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}
    n_images = len(images)

    # Build category -> set of image indices
    cat_ids = sorted(set(c["id"] for c in coco["categories"]))
    cat_to_idx = {c: i for i, c in enumerate(cat_ids)}
    n_cats = len(cat_ids)

    cat_to_images: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        img_idx = id_to_idx.get(ann["image_id"])
        if img_idx is not None:
            cat_to_images[ann["category_id"]].add(img_idx)

    # Binary label matrix: M[image, category] = 1 if image contains category
    M = np.zeros((n_images, n_cats), dtype=np.float64)
    for cat_id, img_indices in cat_to_images.items():
        if cat_id in cat_to_idx:
            for img_idx in img_indices:
                M[img_idx, cat_to_idx[cat_id]] = 1.0

    # Force singleton-category images into train
    forced_train = set()
    singleton_cats = []
    for cat_id in cat_ids:
        img_set = cat_to_images.get(cat_id, set())
        if len(img_set) == 1:
            forced_train.update(img_set)
            singleton_cats.append(cat_id)

    # Remaining images to split
    free_indices = sorted(set(range(n_images)) - forced_train)
    rng.shuffle(free_indices)

    # Iterative stratification on free images
    # Desired ratio for val
    n_free = len(free_indices)
    n_val_target = int(round(n_free * val_fraction))

    # Track per-category counts in each fold
    train_counts = np.zeros(n_cats, dtype=np.float64)
    val_counts = np.zeros(n_cats, dtype=np.float64)

    # Add forced-train images to train counts
    for idx in forced_train:
        train_counts += M[idx]

    # Sort categories by frequency in free images (rarest first)
    free_set = set(free_indices)
    cat_freq_in_free = []
    for ci, cat_id in enumerate(cat_ids):
        freq = sum(1 for idx in cat_to_images.get(cat_id, set()) if idx in free_set)
        cat_freq_in_free.append((freq, ci, cat_id))
    cat_freq_in_free.sort()

    # For each image, compute "rarest category" rank for sorting
    image_rarest_cat_freq = {}
    for idx in free_indices:
        cats_in_image = np.where(M[idx] > 0)[0]
        if len(cats_in_image) == 0:
            image_rarest_cat_freq[idx] = float("inf")
        else:
            freqs = []
            for ci in cats_in_image:
                cat_id = cat_ids[ci]
                freqs.append(len([i for i in cat_to_images.get(cat_id, set()) if i in free_set]))
            image_rarest_cat_freq[idx] = min(freqs)

    # Sort images: those with rarest categories first (ensures rare cats get distributed)
    free_indices_sorted = sorted(free_indices, key=lambda idx: (image_rarest_cat_freq[idx], idx))

    val_set = set()
    train_set = set(forced_train)

    for idx in free_indices_sorted:
        if len(val_set) >= n_val_target:
            train_set.add(idx)
            train_counts += M[idx]
            continue

        if len(train_set) - len(forced_train) >= n_free - n_val_target:
            val_set.add(idx)
            val_counts += M[idx]
            continue

        # Compute imbalance: for each category in this image, which fold needs it more?
        cats_in_image = np.where(M[idx] > 0)[0]
        val_need = 0.0
        train_need = 0.0
        for ci in cats_in_image:
            total = train_counts[ci] + val_counts[ci] + 1  # +1 for this image
            desired_val = total * val_fraction
            desired_train = total * (1 - val_fraction)
            val_need += max(0, desired_val - val_counts[ci])
            train_need += max(0, desired_train - train_counts[ci])

        if val_need >= train_need:
            val_set.add(idx)
            val_counts += M[idx]
        else:
            train_set.add(idx)
            train_counts += M[idx]

    # Build output COCO dicts
    train_image_ids = {image_ids[i] for i in train_set}
    val_image_ids = {image_ids[i] for i in val_set}

    train_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [img for img in images if img["id"] in train_image_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in train_image_ids],
    }
    val_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [img for img in images if img["id"] in val_image_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in val_image_ids],
    }

    # Log stats
    val_cats = set(a["category_id"] for a in val_coco["annotations"])
    train_cats = set(a["category_id"] for a in train_coco["annotations"])
    all_cats = set(c["id"] for c in coco["categories"])
    missing_from_val = all_cats - val_cats

    print(f"Split: {len(train_coco['images'])} train, {len(val_coco['images'])} val "
          f"({len(val_coco['images'])/len(images)*100:.0f}%)")
    print(f"  Train annotations: {len(train_coco['annotations'])}")
    print(f"  Val annotations: {len(val_coco['annotations'])}")
    print(f"  Categories in train: {len(train_cats)}/{len(all_cats)}")
    print(f"  Categories in val: {len(val_cats)}/{len(all_cats)}")
    print(f"  Categories missing from val: {len(missing_from_val)} "
          f"({len(singleton_cats)} are singletons)")

    return train_coco, val_coco


# ---------------------------------------------------------------------------
# Oversampling underrepresented confused categories
# ---------------------------------------------------------------------------

# Pairs where (A, B) means A is frequently misclassified as B.
# Derived from classification_error_analysis.md.
CONFUSION_PAIRS = [
    (253, 224),   # Soft Flora 235G → 540G
    (160, 171),   # Ali Kokmalt → Filtermalt
    (304, 100),   # Evergood Classic Kokmalt → Filtermalt
    (341, 100),   # Evergood Classic Hele Bønner → Filtermalt
    (141, 304),   # Evergood Classic Pressmalt → Kokmalt
    (347, 49),    # Evergood Dark Roast Pressmalt → Filtermalt
    (325, 146),   # Nescafé Gull 100G → 200G
    (292, 137),   # Nescafé Azera Americano → Espresso
    (209, 296),   # Delikatess Sesam → Fiber Balance
    (240, 47),    # Supergranola Glutenfri → Eple&Kanel
    (27, 189),    # Yellow Label 50pos → 25pos
    (18, 175),    # Melange U/Melk → Melange 500G
]


def oversample_confused_categories(
    coco: dict,
    oversample_factor: int = 2,
) -> dict:
    """Duplicate images containing underrepresented variants of confused pairs.

    For each confusion pair (A, B), computes the ratio needed to equalize
    annotation counts, then duplicates images containing the rare variant.
    ``oversample_factor`` caps the maximum duplication (e.g. 4 = at most 4x
    total copies of an image).

    Args:
        coco: COCO dict (train split, pre-patching).
        oversample_factor: Max total copies of an image (2 = at most 2x).

    Returns:
        New COCO dict with duplicated image/annotation entries.
    """
    from collections import Counter

    ann_counts = Counter(a["category_id"] for a in coco["annotations"])

    # For each pair, compute extra copies needed to equalize, capped by factor
    cats_to_oversample: dict[int, int] = {}  # cat_id -> extra copies needed
    for cat_a, cat_b in CONFUSION_PAIRS:
        count_a = ann_counts.get(cat_a, 0)
        count_b = ann_counts.get(cat_b, 0)
        if count_a == 0 and count_b == 0:
            continue
        # Oversample whichever side is smaller
        if count_a < count_b and count_a > 0:
            # Need ratio x copies of A to match B
            ratio = count_b / count_a
            extra = min(int(np.ceil(ratio)) - 1, oversample_factor - 1)
            if extra > 0:
                cats_to_oversample[cat_a] = max(cats_to_oversample.get(cat_a, 0), extra)
        elif count_b < count_a and count_b > 0:
            ratio = count_a / count_b
            extra = min(int(np.ceil(ratio)) - 1, oversample_factor - 1)
            if extra > 0:
                cats_to_oversample[cat_b] = max(cats_to_oversample.get(cat_b, 0), extra)

    if not cats_to_oversample:
        print("Oversampling: no underrepresented categories found, skipping")
        return coco

    # Find images containing each category to oversample
    cat_to_img_ids: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        if ann["category_id"] in cats_to_oversample:
            cat_to_img_ids[ann["category_id"]].add(ann["image_id"])

    # Collect (image_id, extra_copies) — take max if image has multiple cats to oversample
    img_extra: dict[int, int] = {}
    for cat_id, extra in cats_to_oversample.items():
        for img_id in cat_to_img_ids[cat_id]:
            img_extra[img_id] = max(img_extra.get(img_id, 0), extra)

    # Build new COCO dict with duplicated entries
    new_images = list(coco["images"])
    new_annotations = list(coco["annotations"])
    max_img_id = max(img["id"] for img in coco["images"])
    max_ann_id = max(ann["id"] for ann in coco["annotations"])

    img_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    total_dup_images = 0
    total_dup_anns = 0

    for img_id, extra in sorted(img_extra.items()):
        orig_img = img_by_id[img_id]
        orig_anns = anns_by_img[img_id]

        for _ in range(extra):
            max_img_id += 1
            new_img = {**orig_img, "id": max_img_id}
            new_images.append(new_img)
            total_dup_images += 1

            for ann in orig_anns:
                max_ann_id += 1
                new_annotations.append({**ann, "id": max_ann_id, "image_id": max_img_id})
                total_dup_anns += 1

    # Log stats
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    print(f"\nOversampling (factor={oversample_factor}):")
    for cat_id, extra in sorted(cats_to_oversample.items(), key=lambda x: -x[1]):
        n_imgs = len(cat_to_img_ids[cat_id])
        name = cat_names.get(cat_id, str(cat_id))
        print(f"  {name[:45]:<45}  +{extra}x  ({n_imgs} imgs → {n_imgs * (1 + extra)} imgs)")
    print(f"  Total: {len(coco['images'])} → {len(new_images)} images "
          f"(+{total_dup_images}), {len(coco['annotations'])} → {len(new_annotations)} anns "
          f"(+{total_dup_anns})")

    return {
        **coco,
        "images": new_images,
        "annotations": new_annotations,
    }


def _write_no_patch_dataset(coco: dict, output_dir: Path, create_dummy_valid: bool = True):
    """Write a COCO dataset directly (no patching) for rfdetr training.

    Images are referenced by absolute path in the COCO JSON — no copying needed.
    rfdetr's transform pipeline handles resizing to model resolution.
    """
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    with open(train_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)

    if create_dummy_valid:
        (output_dir / "valid").mkdir(parents=True, exist_ok=True)
        first_image = coco["images"][0]
        valid_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": [first_image],
            "annotations": [a for a in coco["annotations"] if a["image_id"] == first_image["id"]],
        }
        with open(output_dir / "valid" / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)

    print(f"No-patch dataset: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations")


def _replace_annotations(
    split_coco: dict,
    alt_coco_path: Path,
    images_dir: Path,
) -> dict:
    """Replace annotations in a split COCO dict with those from an alternative file.

    Keeps the same images (by image_id), but substitutes annotations from
    alt_coco_path. Used to train on cleaned labels while keeping the same
    image split determined by the original annotations.
    """
    with open(alt_coco_path) as f:
        alt_coco = json.load(f)

    split_image_ids = {img["id"] for img in split_coco["images"]}
    alt_anns = [a for a in alt_coco["annotations"] if a["image_id"] in split_image_ids]

    # Build image lookup from alt file (for metadata like width/height)
    alt_img_by_id = {}
    for img in alt_coco["images"]:
        if img["id"] in split_image_ids:
            resolved = img.copy()
            basename = Path(img["file_name"]).name
            resolved["file_name"] = str((images_dir / basename).resolve())
            alt_img_by_id[img["id"]] = resolved

    # Prefer alt images (may have corrected metadata), fall back to split
    new_images = [alt_img_by_id.get(img["id"], img) for img in split_coco["images"]]

    n_orig = len(split_coco["annotations"])
    n_new = len(alt_anns)
    print(f"Replaced train annotations: {n_orig} -> {n_new} (from {alt_coco_path.name})")

    return {
        "info": split_coco.get("info", {}),
        "licenses": split_coco.get("licenses", []),
        "categories": split_coco["categories"],
        "images": new_images,
        "annotations": alt_anns,
    }


def prepare_split_datasets(
    coco_path: Path,
    images_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.5,
    seed: int = 42,
    patch_size: int | None = None,
    min_overlap: int | None = None,
    crop_mode: str = "fixed",
    oversample_factor: int = 1,
    train_coco_path: Path | None = None,
    merge_labels: bool = False,
    max_image_dim: int = 0,
) -> tuple[Path, dict | None]:
    """Split and prepare datasets ready for training.

    Args:
        coco_path: Path to annotations.json (COCO format). Used for splitting
            and for val ground truth.
        images_dir: Directory containing the original full-size images.
        output_dir: Where to write datasets.
        val_fraction: Fraction of images for validation (0 = train on all).
        seed: Random seed.
        patch_size: Patch size for patching (defaults to config.PATCH_SIZE).
        min_overlap: Min overlap for patching (defaults to config.MIN_OVERLAP).
        crop_mode: "fixed" = pre-generate overlapping patches,
            "resize" = use full images (rfdetr resizes),
            "random" = use full images (RandomCropDataset does on-the-fly crops).
        oversample_factor: Duplicate images with underrepresented confused
            categories up to this many times (1 = no oversampling).
        train_coco_path: Optional alternative annotations for training only.
            If set, train split annotations are replaced with these (e.g.
            cleaned labels). Val split always uses coco_path.
        merge_labels: Merge variant category IDs into canonical ones
            (see LABEL_MERGE_MAP).

    Returns:
        (dataset_dir, val_coco_full) where dataset_dir has train/ and valid/
        subdirectories for rfdetr, and val_coco_full is the un-patched val COCO
        dict (None if val_fraction=0).
    """
    if patch_size is None:
        patch_size = PATCH_SIZE
    if min_overlap is None:
        min_overlap = MIN_OVERLAP

    # Check cache (with file lock to prevent races between parallel runs)
    config_str = f"{coco_path}:{train_coco_path}:{val_fraction}:{seed}:{patch_size}:{min_overlap}:{crop_mode}:{oversample_factor}:{sorted(EXCLUDED_IMAGE_IDS)}:{merge_labels}:{max_image_dim}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    cache_file = output_dir / f".split_hash_{config_hash}"

    # Acquire exclusive lock so parallel runs with the same config don't race.
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = output_dir / ".split_lock"
    _lock_fh = open(lock_file, "w")
    fcntl.flock(_lock_fh, fcntl.LOCK_EX)

    try:
        if cache_file.exists() and (output_dir / "train" / "_annotations.coco.json").exists():
            print(f"Using cached split at {output_dir} (hash {config_hash})")
            val_json = output_dir / "val_full.json"
            if val_json.exists():
                with open(val_json) as f:
                    val_coco = json.load(f)
                return output_dir, val_coco
            return output_dir, None

        # Load and resolve image paths
        with open(coco_path) as f:
            coco = json.load(f)

        for img in coco["images"]:
            basename = Path(img["file_name"]).name
            img["file_name"] = str((images_dir / basename).resolve())

        # Remove images with irreparably bad annotations
        coco = _exclude_images(coco)

        # Merge variant categories into canonical ones
        if merge_labels:
            coco = apply_label_merges(coco)

        if val_fraction <= 0:
            # Full training mode — no split
            print("Full training mode: using all images for training")
            if crop_mode == "fixed":
                generate_patched_coco(
                    coco, output_dir,
                    patch_size=patch_size, min_overlap=min_overlap,
                    subdir="train", create_dummy_valid=True,
                    max_image_dim=max_image_dim,
                )
            else:
                # "resize" and "random" both write full images (no patching on disk)
                _write_no_patch_dataset(coco, output_dir)
            cache_file.write_text(config_str)
            return output_dir, None

        # Split
        train_coco, val_coco = iterative_stratification(coco, val_fraction, seed)

        # Replace train annotations with cleaned labels if specified
        if train_coco_path is not None:
            train_coco = _replace_annotations(train_coco, train_coco_path, images_dir)

        # Oversample underrepresented confused categories
        if oversample_factor > 1:
            train_coco = oversample_confused_categories(train_coco, oversample_factor)

        # Prepare train data
        if crop_mode == "fixed":
            print(f"\nPatching train split...")
            generate_patched_coco(
                train_coco, output_dir,
                patch_size=patch_size, min_overlap=min_overlap,
                subdir="train", create_dummy_valid=True,
                max_image_dim=max_image_dim,
            )
        else:
            print(f"\nWriting train split (crop_mode={crop_mode})...")
            _write_no_patch_dataset(train_coco, output_dir)

        # Save un-patched val COCO (for full-image competition eval)
        val_json = output_dir / "val_full.json"
        with open(val_json, "w") as f:
            json.dump(val_coco, f)
        print(f"Saved un-patched val COCO: {val_json} ({len(val_coco['images'])} images)")

        # Write cache marker
        for old in output_dir.glob(".split_hash_*"):
            old.unlink()
        cache_file.write_text(config_str)

        return output_dir, val_coco
    finally:
        fcntl.flock(_lock_fh, fcntl.LOCK_UN)
        _lock_fh.close()
