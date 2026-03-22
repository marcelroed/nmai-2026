"""Visualize batches from the GPU loader with bounding boxes.

Usage:
    from norgesgruppen.dali_loader import GpuPatchLoader
    from norgesgruppen.visualize import show_batch, save_batch

    loader = GpuPatchLoader(coco_json, batch_size=8, augment=True)
    images, targets = next(iter(loader))

    show_batch(images, targets)           # display inline (notebook)
    save_batch(images, targets, "out.png")  # save to file
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

MEANS = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STDS = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Distinct colors for bounding boxes (repeats if > 20 classes in view)
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization. Input [B, 3, H, W] float, output [B, 3, H, W] uint8."""
    imgs = images.cpu().float()
    imgs = imgs * STDS + MEANS
    imgs = (imgs * 255).clamp(0, 255).byte()
    return imgs


def _cxcywh_norm_to_xyxy_pixel(boxes: torch.Tensor, resolution: int) -> torch.Tensor:
    """Convert cxcywh normalized to xyxy pixel coords."""
    if len(boxes) == 0:
        return boxes
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * resolution
    y1 = (cy - h / 2) * resolution
    x2 = (cx + w / 2) * resolution
    y2 = (cy + h / 2) * resolution
    return torch.stack([x1, y1, x2, y2], dim=1)


def draw_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> Image.Image:
    """Draw bounding boxes on an RGB numpy image. Returns PIL Image."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        color = COLORS[i % len(COLORS)] if labels is None else COLORS[int(labels[i]) % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if labels is not None:
            label_id = int(labels[i])
            text = class_names[label_id] if class_names and label_id < len(class_names) else str(label_id)
            # Truncate long names
            if len(text) > 20:
                text = text[:18] + ".."
            draw.text((x1 + 2, max(0, y1 - 13)), text, fill=color, font=font)

    return pil


def render_batch(
    images: torch.Tensor,
    targets: list[dict],
    class_names: list[str] | None = None,
    max_images: int = 16,
    cols: int = 4,
) -> Image.Image:
    """Render a grid of images with bounding boxes.

    Args:
        images: [B, 3, H, W] normalized tensor (GPU or CPU).
        targets: list of dicts with "boxes" (cxcywh norm) and "labels".
        class_names: optional list mapping label id -> name.
        max_images: max number of images to show.
        cols: number of columns in the grid.

    Returns:
        PIL Image of the grid.
    """
    n = min(len(images), max_images)
    resolution = images.shape[-1]

    imgs_uint8 = _denormalize(images[:n])

    panels = []
    for i in range(n):
        img_np = imgs_uint8[i].permute(1, 2, 0).numpy()  # HWC
        boxes = targets[i]["boxes"].cpu()
        labels = targets[i]["labels"].cpu().numpy()
        boxes_xyxy = _cxcywh_norm_to_xyxy_pixel(boxes, resolution).numpy()

        panel = draw_boxes(img_np, boxes_xyxy, labels, class_names)
        panels.append(panel)

    # Arrange in grid
    rows = (n + cols - 1) // cols
    pw, ph = panels[0].size
    grid = Image.new("RGB", (cols * pw, rows * ph), (40, 40, 40))

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        grid.paste(panel, (c * pw, r * ph))

    return grid


def save_batch(
    images: torch.Tensor,
    targets: list[dict],
    path: str | Path = "batch_vis.png",
    class_names: list[str] | None = None,
    **kwargs,
):
    """Render and save a batch visualization to a file."""
    grid = render_batch(images, targets, class_names=class_names, **kwargs)
    grid.save(str(path))
    print(f"Saved batch visualization to {path}")


def show_batch(
    images: torch.Tensor,
    targets: list[dict],
    class_names: list[str] | None = None,
    **kwargs,
):
    """Render and display a batch (works in notebooks and IPython)."""
    grid = render_batch(images, targets, class_names=class_names, **kwargs)
    try:
        from IPython.display import display
        display(grid)
    except ImportError:
        # Fallback: save to temp file
        save_batch(images, targets, "batch_vis.png", class_names=class_names, **kwargs)
