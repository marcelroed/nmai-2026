"""Test train/val splitting logic."""

import json
import tempfile
from pathlib import Path

from norgesgruppen.split import split_coco, write_val_split


def _make_coco(n_images: int = 20, n_anns_per_image: int = 5) -> dict:
    """Create a minimal COCO dict for testing."""
    images = [
        {"id": i, "file_name": f"img_{i:05d}.jpg", "width": 3000, "height": 3000}
        for i in range(1, n_images + 1)
    ]
    annotations = []
    ann_id = 1
    for img in images:
        for _ in range(n_anns_per_image):
            annotations.append({
                "id": ann_id,
                "image_id": img["id"],
                "category_id": ann_id % 10,
                "bbox": [100, 100, 50, 50],
                "area": 2500,
                "iscrowd": 0,
            })
            ann_id += 1
    categories = [{"id": i, "name": f"cat_{i}", "supercategory": "product"} for i in range(10)]
    return {"images": images, "annotations": annotations, "categories": categories}


def test_split_disjoint():
    """Train and val should have no overlapping images."""
    coco = _make_coco(n_images=50)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        img_dir = Path(tmpdir)
        for img in coco["images"]:
            (img_dir / img["file_name"]).touch()

        train_coco, val_coco = split_coco(coco, image_root=img_dir, val_ratio=0.2, seed=42)

    train_ids = {img["id"] for img in train_coco["images"]}
    val_ids = {img["id"] for img in val_coco["images"]}

    assert len(train_ids & val_ids) == 0, "Train and val share images"
    assert train_ids | val_ids == {img["id"] for img in coco["images"]}, "Not all images assigned"


def test_split_ratio():
    """Val set size should match the requested ratio."""
    coco = _make_coco(n_images=100)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir)
        for img in coco["images"]:
            (img_dir / img["file_name"]).touch()

        train_coco, val_coco = split_coco(coco, image_root=img_dir, val_ratio=0.1, seed=42)

    assert len(val_coco["images"]) == 10
    assert len(train_coco["images"]) == 90


def test_split_annotations_follow_images():
    """All annotations should follow their image to the correct split."""
    coco = _make_coco(n_images=20, n_anns_per_image=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir)
        for img in coco["images"]:
            (img_dir / img["file_name"]).touch()

        train_coco, val_coco = split_coco(coco, image_root=img_dir, val_ratio=0.2, seed=42)

    train_img_ids = {img["id"] for img in train_coco["images"]}
    val_img_ids = {img["id"] for img in val_coco["images"]}

    for ann in train_coco["annotations"]:
        assert ann["image_id"] in train_img_ids
    for ann in val_coco["annotations"]:
        assert ann["image_id"] in val_img_ids

    # Total annotations should be preserved
    assert len(train_coco["annotations"]) + len(val_coco["annotations"]) == len(coco["annotations"])


def test_split_deterministic():
    """Same seed should produce the same split."""
    coco = _make_coco(n_images=50)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir)
        for img in coco["images"]:
            (img_dir / img["file_name"]).touch()

        t1, v1 = split_coco(coco, image_root=img_dir, val_ratio=0.2, seed=42)
        t2, v2 = split_coco(coco, image_root=img_dir, val_ratio=0.2, seed=42)

    assert [img["id"] for img in t1["images"]] == [img["id"] for img in t2["images"]]
    assert [img["id"] for img in v1["images"]] == [img["id"] for img in v2["images"]]


def test_split_different_seeds():
    """Different seeds should (almost certainly) produce different splits."""
    coco = _make_coco(n_images=50)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir)
        for img in coco["images"]:
            (img_dir / img["file_name"]).touch()

        _, v1 = split_coco(coco, image_root=img_dir, val_ratio=0.2, seed=42)
        _, v2 = split_coco(coco, image_root=img_dir, val_ratio=0.2, seed=99)

    ids1 = {img["id"] for img in v1["images"]}
    ids2 = {img["id"] for img in v2["images"]}
    assert ids1 != ids2


def test_split_absolute_paths():
    """File paths should be resolved to absolute."""
    coco = _make_coco(n_images=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir)
        for img in coco["images"]:
            (img_dir / img["file_name"]).touch()

        train_coco, val_coco = split_coco(coco, image_root=img_dir, val_ratio=0.2, seed=42)

    for img in train_coco["images"]:
        assert Path(img["file_name"]).is_absolute()
    for img in val_coco["images"]:
        assert Path(img["file_name"]).is_absolute()


def test_write_val_split():
    """write_val_split should create symlinks and a valid COCO JSON."""
    coco = _make_coco(n_images=5, n_anns_per_image=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        out_dir = Path(tmpdir) / "dataset"

        # Create dummy image files
        for img in coco["images"]:
            (img_dir / img["file_name"]).write_text("fake")

        # Resolve paths (as split_coco would)
        for img in coco["images"]:
            img["file_name"] = str((img_dir / img["file_name"]).resolve())

        write_val_split(coco, output_dir=out_dir)

        valid_dir = out_dir / "valid"
        assert valid_dir.exists()

        ann_path = valid_dir / "_annotations.coco.json"
        assert ann_path.exists()

        with open(ann_path) as f:
            written = json.load(f)

        assert len(written["images"]) == 5
        assert len(written["annotations"]) == 15

        # Check symlinks exist
        for img in written["images"]:
            assert Path(img["file_name"]).exists()
