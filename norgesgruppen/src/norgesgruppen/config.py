"""Centralized model and training configuration.

Change the MODEL_SIZE here to switch everything at once.
"""

from rfdetr import RFDETR2XLarge, RFDETRLarge, RFDETRMedium, RFDETRNano

# ---------------------------------------------------------------------------
# Pick one: "nano", "medium", "large", "xxlarge"
# ---------------------------------------------------------------------------
MODEL_SIZE = "xxlarge"

# ---------------------------------------------------------------------------
# Per-size settings
# ---------------------------------------------------------------------------
_CONFIGS = {
    "nano": {
        "model_cls": RFDETRNano,
        "resolution": 560,
        "patch_size": 560,
        "min_overlap": 270,
    },
    "medium": {
        "model_cls": RFDETRMedium,
        "resolution": 576,
        "patch_size": 576,
        "min_overlap": 280,
    },
    "large": {
        "model_cls": RFDETRLarge,
        "resolution": 704,
        "patch_size": 704,
        "min_overlap": 340,
    },
    "xxlarge": {
        "model_cls": RFDETR2XLarge,
        "resolution": 880,
        "patch_size": 880,
        "min_overlap": 440,
    },
}

_cfg = _CONFIGS[MODEL_SIZE]

MODEL_CLS = _cfg["model_cls"]
RESOLUTION = _cfg["resolution"]
PATCH_SIZE = _cfg["patch_size"]
MIN_OVERLAP = _cfg["min_overlap"]

NUM_CLASSES = 357  # 356 product categories (0-indexed)

# ---------------------------------------------------------------------------
# Max image dimension — images larger than this are downscaled before
# patching (training) and inference. 0 = no limit.
# ---------------------------------------------------------------------------
MAX_IMAGE_DIM: int = 0  # set via --max-image-dim CLI flag

def downscale_if_needed(image, max_dim: int = 0):
    """Downscale a PIL Image so its largest dimension is at most max_dim.

    Returns (image, scale_factor). scale_factor is 1.0 if no resize was needed.
    """
    from PIL import Image as _PILImage
    if max_dim <= 0:
        return image, 1.0
    w, h = image.size
    if max(w, h) <= max_dim:
        return image, 1.0
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), _PILImage.LANCZOS), scale


# ---------------------------------------------------------------------------
# Images to exclude from all training and validation (e.g. bad labels, duplicates)
# Add image IDs (integers) here — they will be removed before any split.
# ---------------------------------------------------------------------------
EXCLUDED_IMAGE_IDS: set[str] = {"img_00295.jpg"}
