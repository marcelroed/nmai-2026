"""Fine-tune RF-DETR on the Norgesgruppen product detection dataset."""

from pathlib import Path as _Path
if _Path(".env").exists():
    import dotenv
    dotenv.load_dotenv()

from pathlib import Path
from typing import Annotated

import typer

from norgesgruppen.splitting import prepare_split_datasets
from norgesgruppen.training import train as train_custom


def main(
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-4,
    output_dir: Annotated[str, typer.Option(help="Output directory for checkpoints")] = "output-xxlarge5",
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size per GPU")] = 4,
    grad_accum: Annotated[int, typer.Option(help="Gradient accumulation steps")] = 1,
    val_fraction: Annotated[float, typer.Option(help="Stratified image-level val fraction (0.5 = 50%%)")] = 0.5,
    seed: Annotated[int, typer.Option(help="Random seed for stratified split")] = 42,
    full_train: Annotated[bool, typer.Option(help="Train on all data (no val split) for final submission")] = False,
    competition_eval_interval: Annotated[int, typer.Option(help="Full-image competition eval every N epochs")] = 5,
    model_size: Annotated[str, typer.Option(help="Model size: 'large' or 'xxlarge'")] = "xxlarge",
    crop_mode: Annotated[str, typer.Option(help="Training crop strategy: 'fixed' (pre-patched), 'random' (on-the-fly crops), 'resize' (full image resize)")] = "fixed",
    oversample: Annotated[int, typer.Option(help="Oversample factor for underrepresented confused categories (1=off)")] = 1,
    class_weight_loss: Annotated[bool, typer.Option(help="Enable per-class loss reweighting (sqrt inverse frequency)")] = False,
    train_annotations: Annotated[str, typer.Option(help="Annotations file for TRAINING (cleaned labels). Empty = same as eval.")] = "",
    eval_annotations: Annotated[str, typer.Option(help="Annotations file for EVALUATION and splitting (original/messy labels).")] = "data/train/annotations.json",
    run_name: Annotated[str, typer.Option(help="Descriptive wandb run name (e.g. 'baseline', 'random-crops')")] = "",
    copy_paste: Annotated[bool, typer.Option(help="Enable copy-paste augmentation (requires crop-mode=random)")] = False,
    merge_labels: Annotated[bool, typer.Option(help="Merge variant category IDs into canonical ones (MÜSLI variants)")] = False,
    max_image_dim: Annotated[int, typer.Option(help="Downscale images so largest dimension <= this (0=no limit)")] = 0,
    crop_scale_min: Annotated[float, typer.Option(help="Min scale for random crops (default 0.5)")] = 0.5,
    crop_scale_max: Annotated[float, typer.Option(help="Max scale for random crops (default 1.5)")] = 1.5,
    synthetic_data: Annotated[str, typer.Option(help="Path to synthetic data dir with annotations.json (empty=disabled)")] = "",
    synthetic_weight: Annotated[float, typer.Option(help="Probability of sampling synthetic vs real (0.5=50/50)")] = 0.0,
):
    from norgesgruppen.training import _MODEL_ARCH
    arch = _MODEL_ARCH[model_size]
    ps = arch["resolution"]
    mo = ps // 2

    if full_train:
        val_fraction = 0.0

    eval_coco_path = Path(eval_annotations)
    train_coco_path = Path(train_annotations) if train_annotations else None
    images_dir = Path("data/train/images")

    # Each unique config gets its own directory so parallel runs never collide.
    import hashlib
    from norgesgruppen.splitting import EXCLUDED_IMAGE_IDS
    config_str = f"{eval_coco_path}:{train_coco_path}:{val_fraction}:{seed}:{ps}:{mo}:{crop_mode}:{oversample}:{sorted(EXCLUDED_IMAGE_IDS)}:{merge_labels}:{max_image_dim}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    split_dir = Path(f"data/split-{config_hash}")

    dataset_dir, val_coco = prepare_split_datasets(
        coco_path=eval_coco_path,
        images_dir=images_dir,
        output_dir=split_dir,
        val_fraction=val_fraction,
        seed=seed,
        patch_size=ps,
        min_overlap=mo,
        crop_mode=crop_mode,
        oversample_factor=oversample,
        train_coco_path=train_coco_path,
        merge_labels=merge_labels,
        max_image_dim=max_image_dim,
    )

    train_custom(
        dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
        lr=lr,
        val_coco=val_coco,
        val_images_dir=images_dir,
        competition_eval_interval=competition_eval_interval,
        model_size=model_size,
        run_name=run_name,
        crop_mode=crop_mode,
        class_weight_loss=class_weight_loss,
        copy_paste=copy_paste,
        merge_labels=merge_labels,
        max_image_dim=max_image_dim,
        crop_scale_min=crop_scale_min,
        crop_scale_max=crop_scale_max,
        synthetic_data=synthetic_data,
        synthetic_weight=synthetic_weight,
    )


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
