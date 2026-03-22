"""Random crop dataset for training — samples random crops from full-size images.

Instead of pre-generating fixed patches, this dataset samples a random scale
and crop position each time, providing much greater data diversity and reducing
overfitting to fixed patch positions.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import Normalize as TVNormalize

from norgesgruppen.augmentation import AUG_CONFIG
from norgesgruppen.patching import crop_boxes

# ImageNet normalization (same as rfdetr)
_NORMALIZE = TVNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def _build_augmentation(aug_config: dict | None = None) -> A.Compose | None:
    """Build Albumentations augmentation pipeline from config.

    aug_config is a dict of {TransformName: {kwargs}} (see augmentation.py).
    """
    if aug_config is None:
        aug_config = AUG_CONFIG

    transforms = []
    for name, params in aug_config.items():
        cls = getattr(A, name, None)
        if cls is not None:
            transforms.append(cls(**params))

    if not transforms:
        return None

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, w, h]
            min_visibility=0.3,
            label_fields=["category_ids"],
        ),
    )


class RandomCropDataset(Dataset):
    """Dataset that samples random scaled crops from full-size images.

    Each __getitem__ call:
    1. Picks a random image (biased toward confused-pair images when copy-paste is on)
    2. Applies a random scale factor
    3. Pads by resolution//2 on all sides (so boundary objects can be centered)
    4. Samples a random resolution×resolution crop
    5. Clips annotations to the crop
    5.5. Copy-paste augmentation (optional): pastes instances of confused/rare categories
    6. Applies augmentations
    7. Returns (tensor, target) in rfdetr-compatible format
    """

    def __init__(
        self,
        coco: dict,
        resolution: int = 880,
        scale_range: tuple[float, float] = (0.5, 1.5),
        samples_per_epoch: int = 5000,
        min_visible_fraction: float = 0.6,
        aug_config: list[dict] | None = None,
        copy_paste: bool = False,
        max_image_dim: int = 0,
    ):
        self.resolution = resolution
        self.scale_range = scale_range
        self.max_image_dim = max_image_dim
        self.samples_per_epoch = samples_per_epoch
        self.min_visible_fraction = min_visible_fraction
        self.pad = resolution // 2
        self.copy_paste_enabled = copy_paste

        # Index images and annotations
        self.images = coco["images"]
        self.anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        # Build augmentation pipeline
        self.aug = _build_augmentation(aug_config)

        # Pre-load all images into RAM (only ~882MB for 249 images)
        self._image_cache: dict[str, Image.Image] = {}
        for img_info in self.images:
            path = img_info["file_name"]
            self._image_cache[path] = Image.open(path).convert("RGB")

        # Copy-paste: build instance bank and confusion image index
        self.instance_bank = None
        self.confusion_image_indices: list[int] = []
        if copy_paste:
            from norgesgruppen.copy_paste import InstanceBank
            from norgesgruppen.splitting import CONFUSION_PAIRS

            self.instance_bank = InstanceBank(coco)

            # Build set of all category IDs involved in confusion pairs
            confusion_cat_ids = set()
            for a, b in CONFUSION_PAIRS:
                confusion_cat_ids.add(a)
                confusion_cat_ids.add(b)

            # Find image indices containing at least one confused category
            for i, img_info in enumerate(self.images):
                anns = self.anns_by_image.get(img_info["id"], [])
                if any(a["category_id"] in confusion_cat_ids for a in anns):
                    self.confusion_image_indices.append(i)

            print(
                f"Copy-paste: {len(self.confusion_image_indices)}/{len(self.images)} "
                f"images contain confused-pair categories"
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        rng = np.random.default_rng()

        # 1. Pick random image (biased toward confused-pair images when copy-paste is on)
        if self.instance_bank is not None and self.confusion_image_indices and rng.random() < 0.3:
            img_info = self.images[rng.choice(self.confusion_image_indices)]
        else:
            img_info = self.images[rng.integers(len(self.images))]
        img_path = img_info["file_name"]
        image = self._image_cache[img_path].copy()

        # Downscale large images before cropping
        from norgesgruppen.config import downscale_if_needed
        image, prescale = downscale_if_needed(image, self.max_image_dim)
        orig_w, orig_h = image.size

        # Get annotations for this image
        anns = self.anns_by_image.get(img_info["id"], [])
        if anns:
            boxes_xywh = np.array([a["bbox"] for a in anns], dtype=np.float64) * prescale
            cat_ids = np.array([a["category_id"] for a in anns], dtype=np.int64)
        else:
            boxes_xywh = np.zeros((0, 4), dtype=np.float64)
            cat_ids = np.zeros((0,), dtype=np.int64)

        # 2. Random scale
        scale = rng.uniform(*self.scale_range)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Scale annotations
        if len(boxes_xywh) > 0:
            boxes_xywh = boxes_xywh.copy()
            boxes_xywh *= scale

        # 3. Pad by resolution//2 on all sides
        padded_w = new_w + 2 * self.pad
        padded_h = new_h + 2 * self.pad
        padded = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))
        padded.paste(image, (self.pad, self.pad))

        # Shift annotation coordinates by padding offset
        if len(boxes_xywh) > 0:
            boxes_xywh[:, 0] += self.pad
            boxes_xywh[:, 1] += self.pad

        # 4. Random crop position
        max_x = max(0, padded_w - self.resolution)
        max_y = max(0, padded_h - self.resolution)
        crop_x = int(rng.integers(0, max_x + 1))
        crop_y = int(rng.integers(0, max_y + 1))

        # Crop image
        crop_img = padded.crop((crop_x, crop_y, crop_x + self.resolution, crop_y + self.resolution))

        # 5. Clip annotations to crop region
        cropped_boxes, cropped_cats = crop_boxes(
            boxes_xywh, cat_ids,
            crop_x, crop_y, self.resolution, self.resolution,
            min_visible_fraction=self.min_visible_fraction,
        )

        # 5.5 Copy-paste augmentation
        if self.instance_bank is not None:
            from norgesgruppen.copy_paste import apply_copy_paste
            img_np, cropped_boxes, cropped_cats = apply_copy_paste(
                np.array(crop_img), cropped_boxes, cropped_cats,
                self.instance_bank, rng,
                resolution=self.resolution,
            )
        else:
            img_np = np.array(crop_img)  # [H, W, 3] uint8

        # 6. Apply augmentations
        if self.aug is not None and len(cropped_boxes) > 0:
            # Albumentations expects [x, y, w, h] in pixel coords
            bboxes = cropped_boxes.tolist()
            cat_id_list = cropped_cats.tolist()

            try:
                augmented = self.aug(
                    image=img_np,
                    bboxes=bboxes,
                    category_ids=cat_id_list,
                )
                img_np = augmented["image"]
                if augmented["bboxes"]:
                    cropped_boxes = np.array(augmented["bboxes"], dtype=np.float64)
                    cropped_cats = np.array(augmented["category_ids"], dtype=np.int64)
                else:
                    cropped_boxes = np.zeros((0, 4), dtype=np.float64)
                    cropped_cats = np.zeros((0,), dtype=np.int64)
            except Exception:
                # Augmentation can fail with degenerate boxes — skip augmentation
                pass

        # 7. Convert to tensor and normalize
        img_tensor = TF.to_tensor(img_np)  # [3, H, W] float32 in [0, 1]
        img_tensor = _NORMALIZE(img_tensor)

        # 8. Convert boxes from xywh to xyxy, then to normalized cxcywh
        n_boxes = len(cropped_boxes)
        if n_boxes > 0:
            x = cropped_boxes[:, 0]
            y = cropped_boxes[:, 1]
            w = cropped_boxes[:, 2]
            h = cropped_boxes[:, 3]

            # xyxy absolute
            x1, y1, x2, y2 = x, y, x + w, y + h

            # cxcywh normalized
            cx = (x1 + x2) / 2 / self.resolution
            cy = (y1 + y2) / 2 / self.resolution
            bw = (x2 - x1) / self.resolution
            bh = (y2 - y1) / self.resolution

            boxes_tensor = torch.tensor(
                np.stack([cx, cy, bw, bh], axis=1), dtype=torch.float32
            )
            labels_tensor = torch.tensor(cropped_cats, dtype=torch.int64)
            area_tensor = torch.tensor(w * h, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros(n_boxes, dtype=torch.int64),
            "orig_size": torch.tensor([self.resolution, self.resolution], dtype=torch.int64),
            "size": torch.tensor([self.resolution, self.resolution], dtype=torch.int64),
        }

        return img_tensor, target
