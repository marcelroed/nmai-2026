"""DALI-based GPU-resident data loader for patches.

Uses DALI for GPU-accelerated JPEG decoding during initial load, then
holds the entire dataset in VRAM for zero-latency batch serving.
Augmentation is done on GPU via Kornia.

With 180GB Blackwell VRAM, all ~21K patches fit comfortably in float32.

Usage:
    loader = GpuPatchLoader(
        coco_json="data/dataset-patched/train/_annotations.coco.json",
        batch_size=8,
        augment=True,
    )
    for epoch in range(n_epochs):
        for images, targets in loader:
            # images: torch.Tensor [B, 3, H, W] on GPU, normalized
            # targets: list of dicts with "boxes" (cxcywh norm) and "labels"
            ...
        loader.reset()
"""

from __future__ import annotations

import json
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import feed_ndarray

from norgesgruppen.config import RESOLUTION

MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]


def build_augmentation() -> K.AugmentationSequential:
    """Build the GPU augmentation pipeline. Operates on normalized images + xyxy boxes."""
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomPerspective(distortion_scale=0.03, p=0.2),
        K.RandomAffine(degrees=2, shear=3, p=0.2),
        K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01, p=0.3),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.1),
        K.RandomBrightness(brightness=(0.9, 1.1), p=0.2),
        data_keys=["input", "bbox_xyxy"],
        same_on_batch=False,
    )


def _decode_all_to_gpu(file_paths: list[str], resolution: int, device_id: int = 0) -> torch.Tensor:
    """Use DALI to decode all JPEG files on GPU and return a single tensor.

    Returns:
        torch.Tensor of shape [N, 3, H, W], float32, normalized, on GPU.
    """
    n = len(file_paths)
    batch_size = min(64, n)

    @pipeline_def(batch_size=batch_size, num_threads=8, device_id=device_id)
    def decode_pipe():
        jpegs, _ = fn.readers.file(
            files=file_paths,
            random_shuffle=False,
            name="reader",
        )
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        # Normalize: to float, /255, subtract mean, divide std
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            mean=[m * 255 for m in MEANS],
            std=[s * 255 for s in STDS],
            output_layout="CHW",
        )
        return images

    pipe = decode_pipe()
    pipe.build()

    # Pre-allocate output tensor on GPU
    result = torch.zeros((n, 3, resolution, resolution), dtype=torch.float32, device=f"cuda:{device_id}")

    pos = 0
    while pos < n:
        outputs = pipe.run()
        images_batch = outputs[0]  # TensorListGPU

        current_batch = len(images_batch)
        for i in range(current_batch):
            if pos + i >= n:
                break
            tensor = images_batch[i]
            # Verify resolution
            shape = tensor.shape()
            if shape[1] != resolution or shape[2] != resolution:
                raise ValueError(
                    f"Patch {file_paths[pos + i]} has size {shape[2]}x{shape[1]}, "
                    f"expected {resolution}x{resolution}"
                )
            feed_ndarray(tensor, result[pos + i])

        pos += current_batch

        if pos % 1000 < batch_size:
            print(f"  Decoded {min(pos, n)}/{n}")

    return result


class GpuPatchLoader:
    """Preloads all patches into GPU memory via DALI and serves shuffled batches.

    Uses DALI for fast GPU JPEG decoding during init, then serves
    batches directly from VRAM with zero I/O.
    """

    def __init__(
        self,
        coco_json: str | Path,
        batch_size: int = 8,
        resolution: int = RESOLUTION,
        device: str = "cuda",
        shuffle: bool = True,
        augment: bool = True,
    ):
        self.batch_size = batch_size
        self.resolution = resolution
        self.device = device
        self.shuffle = shuffle
        self.augment = augment
        self._aug = build_augmentation() if augment else None

        # Parse COCO annotations
        with open(coco_json) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.categories = coco["categories"]

        # Index annotations by image_id
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        n = len(self.images)
        file_paths = [img["file_name"] for img in self.images]

        # Decode all images on GPU via DALI
        print(f"Decoding {n} patches on GPU via DALI...")
        self.gpu_images = _decode_all_to_gpu(file_paths, resolution)

        # Build targets: store boxes as xyxy pixels (for kornia augmentation)
        # Convert to cxcywh normalized only at yield time
        self.targets = []
        for img_info in self.images:
            anns = anns_by_image.get(img_info["id"], [])

            if anns:
                boxes = np.array([a["bbox"] for a in anns], dtype=np.float32)  # xywh
                # Convert xywh -> xyxy pixels
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                labels = np.array([a["category_id"] for a in anns], dtype=np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

            self.targets.append({
                "boxes": torch.from_numpy(boxes).to(device),
                "labels": torch.from_numpy(labels).to(device),
            })

        mem_gb = self.gpu_images.element_size() * self.gpu_images.nelement() / 1e9
        print(f"Preloaded {n} patches ({mem_gb:.1f} GB VRAM)")

        self.n = n
        self._order = np.arange(n)
        self._pos = 0
        if shuffle:
            np.random.shuffle(self._order)

    def reset(self):
        """Reset for a new epoch."""
        self._pos = 0
        if self.shuffle:
            np.random.shuffle(self._order)

    def __len__(self):
        """Number of batches per epoch."""
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self._pos = 0
        if self.shuffle:
            np.random.shuffle(self._order)
        return self

    def __next__(self) -> tuple[torch.Tensor, list[dict]]:
        if self._pos >= self.n:
            raise StopIteration

        end = min(self._pos + self.batch_size, self.n)
        indices = self._order[self._pos:end]
        self._pos = end

        images = self.gpu_images[indices]
        targets = [self.targets[i] for i in indices]

        # Apply augmentation per-image (variable box counts)
        if self.augment and self._aug is not None:
            aug_images = []
            aug_targets = []
            for i in range(len(indices)):
                img = images[i:i+1]  # [1, 3, H, W]
                boxes = targets[i]["boxes"]  # [N, 4] xyxy
                labels = targets[i]["labels"]

                if len(boxes) > 0:
                    aug_img, aug_boxes = self._aug(img, boxes.unsqueeze(0))
                    aug_boxes = aug_boxes.squeeze(0)
                    # Clip to image bounds and filter degenerate boxes
                    aug_boxes = aug_boxes.clamp(min=0, max=self.resolution)
                    valid = (aug_boxes[:, 2] > aug_boxes[:, 0] + 1) & (aug_boxes[:, 3] > aug_boxes[:, 1] + 1)
                    aug_boxes = aug_boxes[valid]
                    labels = labels[valid]
                else:
                    # No boxes — apply only pixel-level augmentation
                    # Pass a dummy box to satisfy data_keys, then discard
                    dummy = torch.zeros(1, 1, 4, device=img.device)
                    aug_img, _ = self._aug(img, dummy)
                    aug_boxes = boxes

                aug_images.append(aug_img.squeeze(0))
                aug_targets.append({"boxes": aug_boxes, "labels": labels})

            images = torch.stack(aug_images)
            targets = aug_targets

        # Convert boxes from xyxy pixels to cxcywh normalized
        out_targets = []
        for t in targets:
            boxes = t["boxes"].clone()
            if len(boxes) > 0:
                # xyxy -> cxcywh
                cx = (boxes[:, 0] + boxes[:, 2]) / 2 / self.resolution
                cy = (boxes[:, 1] + boxes[:, 3]) / 2 / self.resolution
                w = (boxes[:, 2] - boxes[:, 0]) / self.resolution
                h = (boxes[:, 3] - boxes[:, 1]) / self.resolution
                boxes = torch.stack([cx, cy, w, h], dim=1)
            out_targets.append({"boxes": boxes, "labels": t["labels"]})

        return images, out_targets
