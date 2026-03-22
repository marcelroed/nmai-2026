"""Fine-tune YOLOv26 on the Norgesgruppen product detection dataset."""

from pathlib import Path as _Path

if _Path(".env").exists():
    import dotenv

    dotenv.load_dotenv()

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from norgesgruppen.splitting import EXCLUDED_IMAGE_IDS, prepare_split_datasets
from norgesgruppen.yolo_data import build_class_mapping, coco_to_yolo_dataset
from norgesgruppen.yolo_eval import make_competition_eval_callback


def main(
    yolo_model: Annotated[
        str, typer.Option(help="YOLO model variant (e.g. yolo26n.pt, yolo26x.pt)")
    ] = "yolo26x.pt",
    lr: Annotated[float, typer.Option(help="Initial learning rate")] = 0.01,
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 100,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 16,
    imgsz: Annotated[
        int, typer.Option(help="Input image size (should match patch size)")
    ] = 880,
    val_fraction: Annotated[
        float, typer.Option(help="Stratified image-level val fraction")
    ] = 0.5,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    output_dir: Annotated[
        str, typer.Option(help="Output directory for checkpoints")
    ] = "output-yolo",
    run_name: Annotated[str, typer.Option(help="Descriptive wandb run name")] = "",
    competition_eval_interval: Annotated[
        int, typer.Option(help="Full-image competition eval every N epochs")
    ] = 5,
    train_annotations: Annotated[
        str,
        typer.Option(
            help="Annotations file for TRAINING (cleaned labels). Empty = same as eval."
        ),
    ] = "",
    eval_annotations: Annotated[
        str,
        typer.Option(help="Annotations file for EVALUATION and splitting."),
    ] = "data/train/annotations.json",
    merge_labels: Annotated[
        bool,
        typer.Option(
            help="Merge variant category IDs into canonical ones (MÜSLI variants)"
        ),
    ] = False,
    max_image_dim: Annotated[
        int,
        typer.Option(help="Downscale images so largest dimension <= this (0=no limit)"),
    ] = 0,
    full_train: Annotated[
        bool, typer.Option(help="Train on all data (no val split)")
    ] = False,
    crop_mode: Annotated[
        str,
        typer.Option(
            help="Training crop strategy: 'fixed' (pre-patched) or 'random' (on-the-fly crops)"
        ),
    ] = "random",
    crop_scale_min: Annotated[
        float, typer.Option(help="Min scale for random crops")
    ] = 0.5,
    crop_scale_max: Annotated[
        float, typer.Option(help="Max scale for random crops")
    ] = 1.5,
    samples_per_epoch: Annotated[
        int, typer.Option(help="Samples per epoch for random crop mode")
    ] = 5000,
    mosaic: Annotated[
        float, typer.Option(help="Mosaic augmentation probability (fixed mode only)")
    ] = 1.0,
    copy_paste: Annotated[
        float,
        typer.Option(help="Copy-paste augmentation probability (fixed mode only)"),
    ] = 0.0,
    workers: Annotated[int, typer.Option(help="Number of dataloader workers")] = 8,
):
    from ultralytics import YOLO

    if full_train:
        val_fraction = 0.0

    eval_coco_path = Path(eval_annotations)
    train_coco_path = Path(train_annotations) if train_annotations else None
    images_dir = Path("data/train/images")

    # YOLO requires imgsz to be a multiple of the model stride (32).
    # Round up to avoid Ultralytics silently changing it and causing
    # a mismatch with our crop resolution.
    stride = 32
    if imgsz % stride != 0:
        imgsz = ((imgsz + stride - 1) // stride) * stride
        print(f"  Rounded imgsz up to {imgsz} (must be multiple of {stride})")

    min_overlap = imgsz // 2

    # Use "random" crop_mode for splitting so we get full images (not patches)
    split_crop_mode = "random" if crop_mode == "random" else "fixed"

    # Unique directory per config (same hashing as train_rfdetr.py)
    config_str = (
        f"{eval_coco_path}:{train_coco_path}:{val_fraction}:{seed}:"
        f"{imgsz}:{min_overlap}:{split_crop_mode}:1:{sorted(EXCLUDED_IMAGE_IDS)}:"
        f"{merge_labels}:{max_image_dim}"
    )
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    split_dir = Path(f"data/split-{config_hash}")

    # 1. Split dataset
    print("=== Step 1: Splitting dataset ===")
    dataset_dir, val_coco = prepare_split_datasets(
        coco_path=eval_coco_path,
        images_dir=images_dir,
        output_dir=split_dir,
        val_fraction=val_fraction,
        seed=seed,
        patch_size=imgsz,
        min_overlap=min_overlap,
        crop_mode=split_crop_mode,
        train_coco_path=train_coco_path,
        merge_labels=merge_labels,
        max_image_dim=max_image_dim,
    )

    # Build class mapping from the training COCO data
    train_coco = _load_train_coco(dataset_dir, split_crop_mode)
    coco_to_yolo, yolo_to_coco, yolo_names = _build_full_class_mapping(train_coco)

    if crop_mode == "random":
        _train_random_crop(
            yolo_model=yolo_model,
            train_coco=train_coco,
            val_coco=val_coco,
            coco_to_yolo=coco_to_yolo,
            yolo_to_coco=yolo_to_coco,
            yolo_names=yolo_names,
            images_dir=images_dir,
            imgsz=imgsz,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            output_dir=output_dir,
            run_name=run_name,
            competition_eval_interval=competition_eval_interval,
            max_image_dim=max_image_dim,
            crop_scale_min=crop_scale_min,
            crop_scale_max=crop_scale_max,
            samples_per_epoch=samples_per_epoch,
            workers=workers,
        )
    else:
        _train_fixed_patches(
            yolo_model=yolo_model,
            dataset_dir=dataset_dir,
            val_coco=val_coco,
            yolo_to_coco=yolo_to_coco,
            images_dir=images_dir,
            imgsz=imgsz,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            output_dir=output_dir,
            run_name=run_name,
            competition_eval_interval=competition_eval_interval,
            max_image_dim=max_image_dim,
            mosaic=mosaic,
            copy_paste=copy_paste,
            workers=workers,
        )


def _create_dummy_val_dataset(
    yolo_dir: Path,
    train_coco: dict,
    coco_to_yolo: dict[int, int],
    imgsz: int,
) -> None:
    """Create a minimal val dataset with one image so Ultralytics can initialize."""
    val_images_dir = yolo_dir / "images" / "val"
    val_labels_dir = yolo_dir / "labels" / "val"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Also create train dir (even if unused) so data.yaml is valid
    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Use the first training image as a dummy val image
    dummy_img_path = val_images_dir / "dummy.jpg"
    if not dummy_img_path.exists():
        import cv2
        import numpy as np

        # Create a simple black image
        img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        cv2.imwrite(str(dummy_img_path), img)

    # Write an empty label file
    dummy_label_path = val_labels_dir / "dummy.txt"
    if not dummy_label_path.exists():
        dummy_label_path.write_text("")


def _load_train_coco(dataset_dir: Path, crop_mode: str) -> dict:
    """Load the training COCO dict from the split directory."""
    if crop_mode == "fixed":
        coco_path = dataset_dir / "train" / "_annotations.coco.json"
    else:
        coco_path = dataset_dir / "train" / "_annotations.coco.json"
    with open(coco_path) as f:
        return json.load(f)


def _build_full_class_mapping(
    train_coco: dict,
) -> tuple[dict[int, int], dict[int, int], list[str]]:
    """Build class mapping covering all category IDs in both categories list and annotations."""
    cat_name_lookup = {c["id"]: c["name"] for c in train_coco["categories"]}
    ann_cat_ids = {ann["category_id"] for ann in train_coco["annotations"]}
    all_cat_ids = set(cat_name_lookup.keys()) | ann_cat_ids

    full_categories = []
    for cat_id in sorted(all_cat_ids):
        name = cat_name_lookup.get(cat_id, f"class_{cat_id}")
        full_categories.append({"id": cat_id, "name": name})

    return build_class_mapping(full_categories)


def _train_random_crop(
    yolo_model: str,
    train_coco: dict,
    val_coco: dict | None,
    coco_to_yolo: dict[int, int],
    yolo_to_coco: dict[int, int],
    yolo_names: list[str],
    images_dir: Path,
    imgsz: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: str,
    run_name: str,
    competition_eval_interval: int,
    max_image_dim: int,
    crop_scale_min: float,
    crop_scale_max: float,
    samples_per_epoch: int,
    workers: int,
):
    """Train with on-the-fly random crops using a custom Ultralytics trainer."""
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.build import InfiniteDataLoader, build_dataloader
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    from norgesgruppen.yolo_random_crop_dataset import RandomCropYOLODataset

    # Build the random crop training dataset
    print("=== Step 2: Building random crop dataset ===")
    train_dataset = RandomCropYOLODataset(
        coco=train_coco,
        coco_to_yolo_map=coco_to_yolo,
        resolution=imgsz,
        scale_range=(crop_scale_min, crop_scale_max),
        samples_per_epoch=samples_per_epoch,
        max_image_dim=max_image_dim,
    )
    print(
        f"  Random crop dataset: {len(train_coco['images'])} images, "
        f"{samples_per_epoch} samples/epoch, scale {crop_scale_min}-{crop_scale_max}"
    )

    # We need a YOLO dataset on disk for val (Ultralytics requirement).
    # Create a minimal val set with one dummy image so Ultralytics can init.
    import yaml

    yolo_dir = Path(output_dir) / "yolo_dataset"
    _create_dummy_val_dataset(yolo_dir, train_coco, coco_to_yolo, imgsz)

    names_dict = {i: name for i, name in enumerate(yolo_names)}
    data_yaml = yolo_dir / "data.yaml"
    data_config = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",  # not used (custom dataset overrides)
        "val": "images/val",
        "nc": len(yolo_names),
        "names": names_dict,
    }
    with open(data_yaml, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    # Custom trainer that injects our random crop dataset
    class RandomCropTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            if mode == "train":
                return train_dataset
            return super().build_dataset(img_path, mode, batch)

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                dataset = self.build_dataset(dataset_path, mode, batch_size)
                return InfiniteDataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=min(os.cpu_count() or 1, workers),
                    pin_memory=True,
                    collate_fn=dataset.collate_fn,
                    persistent_workers=workers > 0,
                )
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def plot_training_labels(self):
            """Use labels from our random crop dataset for the training labels plot."""
            from ultralytics.utils.plotting import plot_labels

            boxes = np.concatenate([lb["bboxes"] for lb in train_dataset.labels], 0)
            cls = np.concatenate([lb["cls"] for lb in train_dataset.labels], 0)
            plot_labels(
                boxes, cls.squeeze(), names=self.data["names"],
                save_dir=self.save_dir, on_plot=self.on_plot,
            )

    # Load model and set up training
    print(f"=== Step 3: Loading {yolo_model} ===")
    model = YOLO(yolo_model)

    # Register competition eval callback
    if val_coco is not None:
        print("=== Step 4: Registering competition eval callback ===")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        callback = make_competition_eval_callback(
            val_coco=val_coco,
            images_dir=images_dir,
            imgsz=imgsz,
            yolo_to_coco_map=yolo_to_coco,
            eval_interval=competition_eval_interval,
            max_image_dim=max_image_dim,
            output_dir=out_path,
        )
        model.add_callback("on_fit_epoch_end", callback)

    os.environ.setdefault("WANDB_PROJECT", "norgesgruppen-rfdetr")

    print("=== Step 5: Starting training (random crop) ===")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        project=output_dir,
        name=run_name or "yolo_run",
        seed=seed,
        workers=workers,
        amp=True,
        exist_ok=True,
        verbose=True,
        trainer=RandomCropTrainer,
    )


def _train_fixed_patches(
    yolo_model: str,
    dataset_dir: Path,
    val_coco: dict | None,
    yolo_to_coco: dict[int, int],
    images_dir: Path,
    imgsz: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: str,
    run_name: str,
    competition_eval_interval: int,
    max_image_dim: int,
    mosaic: float,
    copy_paste: float,
    workers: int,
):
    """Train on pre-generated fixed patches using standard Ultralytics pipeline."""
    from ultralytics import YOLO

    print("=== Step 2: Converting to YOLO format ===")
    yolo_dir = Path(output_dir) / "yolo_dataset"
    data_yaml, yolo_to_coco_map = coco_to_yolo_dataset(dataset_dir, yolo_dir)

    print(f"=== Step 3: Loading {yolo_model} ===")
    model = YOLO(yolo_model)

    if val_coco is not None:
        print("=== Step 4: Registering competition eval callback ===")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        callback = make_competition_eval_callback(
            val_coco=val_coco,
            images_dir=images_dir,
            imgsz=imgsz,
            yolo_to_coco_map=yolo_to_coco_map,
            eval_interval=competition_eval_interval,
            max_image_dim=max_image_dim,
            output_dir=out_path,
        )
        model.add_callback("on_fit_epoch_end", callback)

    os.environ.setdefault("WANDB_PROJECT", "norgesgruppen-rfdetr")

    print("=== Step 5: Starting training (fixed patches) ===")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        project=output_dir,
        name=run_name or "yolo_run",
        seed=seed,
        workers=workers,
        mosaic=mosaic,
        copy_paste=copy_paste,
        amp=True,
        exist_ok=True,
        verbose=True,
    )


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
