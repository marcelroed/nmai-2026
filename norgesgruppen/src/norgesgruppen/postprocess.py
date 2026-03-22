"""Composable post-inference transforms for detection outputs.

Each transform takes and returns a supervision.Detections object,
so they can be chained freely:

    detections = model.predict(image)
    detections = apply_transforms(detections, [cross_class_nms(0.6), ...])
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import supervision as sv


Transform = Callable[[sv.Detections], sv.Detections]


def apply_transforms(detections: sv.Detections, transforms: list[Transform]) -> sv.Detections:
    """Apply a list of transforms sequentially."""
    for t in transforms:
        detections = t(detections)
    return detections


def cross_class_nms(iou_threshold: float = 0.6) -> Transform:
    """Suppress overlapping boxes regardless of class.

    Keeps the highest-confidence box when two boxes of any class
    overlap above iou_threshold.
    """
    def _apply(detections: sv.Detections) -> sv.Detections:
        if detections.is_empty():
            return detections
        return detections.with_nms(threshold=iou_threshold, class_agnostic=True)
    return _apply


DEFAULT_TRANSFORMS: list[Transform] = [
    cross_class_nms(iou_threshold=0.6),
]
