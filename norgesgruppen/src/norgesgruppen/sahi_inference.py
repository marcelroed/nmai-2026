"""SAHI-based sliced inference for RF-DETR.

Provides an alternative to our custom patching.predict_with_patches,
using SAHI's battle-tested GREEDYNMM merging instead of edge-filtering + NMS.

Usage:
    from norgesgruppen.sahi_inference import predict_with_sahi, RFDETRSahiModel

    sahi_model = RFDETRSahiModel(model, classes, threshold=0.2)
    detections = predict_with_sahi(image, sahi_model)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import supervision as sv
from PIL import Image
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction

if TYPE_CHECKING:
    from rfdetr import RFDETRLarge

PATCH_SIZE = 880
OVERLAP_RATIO = 0.45  # ~400px overlap at 880px patches
CONFIDENCE_THRESHOLD = 0.2


class RFDETRSahiModel(DetectionModel):
    """Wraps RF-DETR for use with SAHI's sliced inference."""

    def __init__(self, model: RFDETRLarge, classes: list[str], threshold: float = CONFIDENCE_THRESHOLD):
        self._rfdetr = model
        self._classes = classes
        self._threshold = threshold
        super().__init__(
            model=model,
            confidence_threshold=threshold,
            device="cuda",
            category_mapping={str(i): name for i, name in enumerate(classes)},
        )

    def load_model(self):
        pass  # Already loaded

    def set_model(self, model, **kwargs):
        self._rfdetr = model

    def unload_model(self):
        pass

    def perform_inference(self, image: np.ndarray):
        """Run RF-DETR prediction on a numpy image."""
        pil_image = Image.fromarray(image)
        import torch
        with torch.amp.autocast("cuda"):
            detections = self._rfdetr.predict(pil_image, threshold=self._threshold)
        self._original_predictions = detections

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ):
        """Convert RF-DETR sv.Detections to SAHI ObjectPrediction list."""
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]
        if full_shape_list is None:
            full_shape_list = [None]

        dets = self._original_predictions
        object_predictions = []

        if dets.is_empty():
            self._object_prediction_list = []
            return

        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i]
            object_predictions.append(
                ObjectPrediction(
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    category_id=int(dets.class_id[i]),
                    category_name=self._classes[int(dets.class_id[i])] if int(dets.class_id[i]) < len(self._classes) else str(dets.class_id[i]),
                    score=float(dets.confidence[i]),
                    shift_amount=shift_amount_list[0],
                    full_shape=full_shape_list[0],
                )
            )

        self._object_prediction_list = object_predictions


def predict_with_sahi(
    image: Image.Image,
    sahi_model: RFDETRSahiModel,
    patch_size: int = PATCH_SIZE,
    overlap_ratio: float = OVERLAP_RATIO,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
) -> sv.Detections:
    """Run SAHI sliced inference and return sv.Detections.

    Args:
        image: Full-size PIL image.
        sahi_model: RFDETRSahiModel instance.
        patch_size: Slice dimensions (square).
        overlap_ratio: Overlap between adjacent slices.
        postprocess_type: "GREEDYNMM", "NMM", or "NMS".
        postprocess_match_metric: "IOU" or "IOS".
        postprocess_match_threshold: Threshold for merging/suppression.

    Returns:
        sv.Detections in full-image coordinates.
    """
    image_np = np.array(image)

    result = get_sliced_prediction(
        image=image_np,
        detection_model=sahi_model,
        slice_height=patch_size,
        slice_width=patch_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=False,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=postprocess_match_threshold,
        postprocess_class_agnostic=True,
        verbose=0,
    )

    object_predictions = result.object_prediction_list
    if not object_predictions:
        return sv.Detections.empty()

    xyxy = np.array([pred.bbox.to_xyxy() for pred in object_predictions])
    confidence = np.array([pred.score.value for pred in object_predictions])
    class_id = np.array([pred.category.id for pred in object_predictions], dtype=int)

    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
