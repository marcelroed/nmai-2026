"""Random crop dataset for YOLO training — samples random crops from full-size images.

Instead of pre-generating fixed patches, this dataset samples a random scale
and crop position each time, providing much greater data diversity.

Returns data in the format expected by Ultralytics' collate_fn:
  - img: (3, H, W) uint8 tensor (BGR, CHW)
  - cls: (N, 1) float tensor of class indices
  - bboxes: (N, 4) float tensor of normalized xywh boxes
  - batch_idx: (N,) float tensor of zeros
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from norgesgruppen.patching import crop_boxes


class RandomCropYOLODataset(Dataset):
    """Dataset that samples random scaled crops from full-size images.

    Each __getitem__ call:
    1. Picks a random image
    2. Applies a random scale factor
    3. Pads by resolution//2 on all sides
    4. Samples a random resolution x resolution crop
    5. Clips annotations to the crop
    6. Returns (img, cls, bboxes, batch_idx) in Ultralytics format
    """

    def __init__(
        self,
        coco: dict,
        coco_to_yolo_map: dict[int, int],
        resolution: int = 880,
        scale_range: tuple[float, float] = (0.5, 1.5),
        samples_per_epoch: int = 5000,
        min_visible_fraction: float = 0.6,
        max_image_dim: int = 0,
    ):
        self.resolution = resolution
        self.scale_range = scale_range
        self.samples_per_epoch = samples_per_epoch
        self.min_visible_fraction = min_visible_fraction
        self.max_image_dim = max_image_dim
        self.pad = resolution // 2
        self.coco_to_yolo_map = coco_to_yolo_map

        # Index images and annotations
        self.images = coco["images"]
        self.anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        # Pre-load all images into RAM as BGR numpy arrays (OpenCV format)
        self._image_cache: dict[str, np.ndarray] = {}
        for img_info in self.images:
            path = img_info["file_name"]
            img = cv2.imread(path)  # BGR
            if img is None:
                img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
            self._image_cache[path] = img

        # Build labels list for compatibility with Ultralytics trainer
        # (used by plot_training_labels, etc.)
        self.labels = []
        for img_info in self.images:
            anns = self.anns_by_image.get(img_info["id"], [])
            if anns:
                img_w = img_info["width"]
                img_h = img_info["height"]
                cls = np.array(
                    [[self.coco_to_yolo_map[a["category_id"]]] for a in anns],
                    dtype=np.float32,
                )
                bboxes = np.array(
                    [
                        [
                            (a["bbox"][0] + a["bbox"][2] / 2) / img_w,
                            (a["bbox"][1] + a["bbox"][3] / 2) / img_h,
                            a["bbox"][2] / img_w,
                            a["bbox"][3] / img_h,
                        ]
                        for a in anns
                    ],
                    dtype=np.float32,
                )
            else:
                cls = np.zeros((0, 1), dtype=np.float32)
                bboxes = np.zeros((0, 4), dtype=np.float32)
            self.labels.append({"cls": cls, "bboxes": bboxes})

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate samples into a batch (same format as YOLODataset.collate_fn)."""
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            elif k in {"bboxes", "cls"}:
                value = torch.cat(value, 0)
            elif k in {"im_file", "ori_shape", "resized_shape"}:
                value = list(value)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng()

        # 1. Pick random image
        img_info = self.images[rng.integers(len(self.images))]
        img_path = img_info["file_name"]
        image = self._image_cache[img_path].copy()  # BGR HWC

        # Downscale large images
        h0, w0 = image.shape[:2]
        prescale = 1.0
        if self.max_image_dim > 0 and max(h0, w0) > self.max_image_dim:
            prescale = self.max_image_dim / max(h0, w0)
            new_w, new_h = int(w0 * prescale), int(h0 * prescale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        orig_h, orig_w = image.shape[:2]

        # Get annotations
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
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if len(boxes_xywh) > 0:
            boxes_xywh = boxes_xywh.copy()
            boxes_xywh *= scale

        # 3. Pad by resolution//2 on all sides
        padded = cv2.copyMakeBorder(
            image,
            self.pad, self.pad, self.pad, self.pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        if len(boxes_xywh) > 0:
            boxes_xywh[:, 0] += self.pad
            boxes_xywh[:, 1] += self.pad

        # 4. Random crop
        padded_h, padded_w = padded.shape[:2]
        max_x = max(0, padded_w - self.resolution)
        max_y = max(0, padded_h - self.resolution)
        crop_x = int(rng.integers(0, max_x + 1))
        crop_y = int(rng.integers(0, max_y + 1))

        crop_img = padded[
            crop_y : crop_y + self.resolution,
            crop_x : crop_x + self.resolution,
        ]

        # Ensure correct size (pad if at edge)
        ch, cw = crop_img.shape[:2]
        if ch < self.resolution or cw < self.resolution:
            padded_crop = np.zeros(
                (self.resolution, self.resolution, 3), dtype=np.uint8
            )
            padded_crop[:ch, :cw] = crop_img
            crop_img = padded_crop

        # 5. Clip annotations to crop
        cropped_boxes, cropped_cats = crop_boxes(
            boxes_xywh,
            cat_ids,
            crop_x,
            crop_y,
            self.resolution,
            self.resolution,
            min_visible_fraction=self.min_visible_fraction,
        )

        # 6. Convert to YOLO format
        n = len(cropped_boxes)
        if n > 0:
            x, y, w, h = (
                cropped_boxes[:, 0],
                cropped_boxes[:, 1],
                cropped_boxes[:, 2],
                cropped_boxes[:, 3],
            )
            # Normalized center xywh
            cx = (x + w / 2) / self.resolution
            cy = (y + h / 2) / self.resolution
            nw = w / self.resolution
            nh = h / self.resolution

            bboxes = torch.tensor(
                np.stack([cx, cy, nw, nh], axis=1), dtype=torch.float32
            )
            # Map COCO category IDs to YOLO contiguous indices
            cls = torch.tensor(
                [[self.coco_to_yolo_map[int(c)]] for c in cropped_cats],
                dtype=torch.float32,
            )
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            cls = torch.zeros((0, 1), dtype=torch.float32)

        # Convert image: BGR HWC uint8 -> CHW tensor
        img_tensor = torch.from_numpy(crop_img.transpose(2, 0, 1).copy())

        return {
            "img": img_tensor,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.zeros(n, dtype=torch.float32),
            "im_file": img_path,
            "ori_shape": (self.resolution, self.resolution),
            "resized_shape": (self.resolution, self.resolution),
        }
