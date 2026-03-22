"""Augmentation config for RF-DETR training.

RF-DETR accepts an Albumentations config dict via the `aug_config` parameter.
Each key is an Albumentations transform class name, and the value is a dict
of kwargs passed to that transform's constructor.
"""

AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "Perspective": {"scale": (0.02, 0.05), "p": 0.2},
    "Affine": {"shear": (-3, 3), "rotate": (-2, 2), "p": 0.2},
    "ColorJitter": {"brightness": 0.25, "contrast": 0.25, "saturation": 0.2, "hue": 0.02, "p": 0.5},
    "GaussianBlur": {"blur_limit": (3, 5), "p": 0.15},
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.3},
}
