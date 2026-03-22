"""Generate synthetic shelf images from individual product reference photos.

Places product reference images on varied random backgrounds in dense
shelf arrangements, with per-product perspective transforms, to create
training data with:
- Many products per image (40-120, matching real data density)
- Variable image resolution (880×880 to 3000×3000)
- Hard pairs placed side-by-side
- Per-product perspective/rotation transforms
- Diverse random backgrounds (procedural textures, gradients, noise patterns)

Output: COCO-format annotations + images in a specified directory.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from norgesgruppen.splitting import CONFUSION_PAIRS

# ---------------------------------------------------------------------------
# Per-product augmentation (applied individually before placement)
# Stronger perspective transforms as requested
# ---------------------------------------------------------------------------
PRODUCT_AUG = A.Compose([
    A.Perspective(scale=(0.02, 0.06), p=0.5),
    A.Affine(shear=(-5, 5), rotate=(-5, 5), p=0.4),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.25),
])

# Whole-image augmentation (after composition)
IMAGE_AUG = A.Compose([
    A.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.01, p=0.3),
    A.GaussNoise(var_limit=(2, 8), p=0.1),
])

ALL_VIEWS = ["front", "main", "left", "right", "back", "top", "bottom"]

# ---------------------------------------------------------------------------
# Random background generators
# ---------------------------------------------------------------------------


def _bg_solid_noisy(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Solid color with heavy per-pixel noise for texture."""
    r, g, b = rng.randint(30, 240), rng.randint(30, 240), rng.randint(30, 240)
    canvas = np.full((h, w, 3), (r, g, b), dtype=np.uint8)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    noise = nrng.integers(-25, 26, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _bg_gradient_multi(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Multi-stop gradient with noise overlay."""
    n_stops = rng.randint(2, 5)
    colors = [np.array([rng.randint(20, 240) for _ in range(3)], dtype=np.float32) for _ in range(n_stops)]
    vertical = rng.random() < 0.5

    length = h if vertical else w
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(n_stops - 1):
        start = int(length * i / (n_stops - 1))
        end = int(length * (i + 1) / (n_stops - 1))
        for j in range(start, end):
            t = (j - start) / max(1, end - start)
            color = colors[i] * (1 - t) + colors[i + 1] * t
            if vertical:
                canvas[j, :] = color
            else:
                canvas[:, j] = color

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    # Add grain
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    noise = nrng.integers(-12, 13, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _bg_wood_like(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Wood-grain-like texture using sine waves + noise."""
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    # Base color (brown/tan/gray range)
    base = np.array([rng.randint(100, 200), rng.randint(80, 170), rng.randint(50, 140)], dtype=np.float32)
    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)

    freq = rng.uniform(0.005, 0.02)
    phase = rng.uniform(0, 2 * math.pi)
    wave = np.sin(y_coords * freq + x_coords * freq * 0.3 + phase) * 30
    wave += np.sin(y_coords * freq * 3.7 + phase * 2) * 10

    canvas = np.clip(base + wave[:, :, np.newaxis], 0, 255).astype(np.uint8)
    # Add fine grain noise
    noise = nrng.integers(-15, 16, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _bg_fabric_like(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Woven fabric-like texture using overlapping fine lines."""
    base_color = np.array([rng.randint(60, 220) for _ in range(3)], dtype=np.uint8)
    canvas = np.full((h, w, 3), base_color, dtype=np.uint8)

    line_color1 = np.array([max(0, c - rng.randint(20, 60)) for c in base_color], dtype=np.uint8)
    line_color2 = np.array([min(255, c + rng.randint(10, 40)) for c in base_color], dtype=np.uint8)
    spacing = rng.randint(4, 12)

    for y in range(0, h, spacing):
        canvas[y:y + 1, :] = line_color1
    for x in range(0, w, spacing):
        canvas[:, x:x + 1] = line_color2

    # Slight blur for softness
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    noise = nrng.integers(-8, 9, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _bg_concrete_like(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Concrete/stone-like speckled texture."""
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    base_val = rng.randint(100, 200)
    canvas = nrng.integers(base_val - 30, base_val + 30, size=(h, w), dtype=np.uint8)
    # Multi-scale noise
    small = nrng.integers(base_val - 40, base_val + 40, size=(h // 8, w // 8), dtype=np.uint8)
    large_noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    canvas = np.clip(canvas.astype(np.int16) + large_noise.astype(np.int16) - base_val, 0, 255).astype(np.uint8)
    # Tint with a random color
    tint = np.array([rng.randint(180, 255), rng.randint(170, 250), rng.randint(160, 240)], dtype=np.float32) / 255.0
    canvas_3 = (canvas[:, :, np.newaxis].astype(np.float32) * tint).astype(np.uint8)
    return canvas_3


def _bg_multi_color_blocks(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Overlapping random colored rectangles with texture."""
    base = np.array([rng.randint(60, 200) for _ in range(3)], dtype=np.uint8)
    canvas = np.full((h, w, 3), base, dtype=np.uint8)
    n_blocks = rng.randint(8, 30)
    for _ in range(n_blocks):
        color = [rng.randint(20, 245) for _ in range(3)]
        bx = rng.randint(0, w)
        by = rng.randint(0, h)
        bw = rng.randint(w // 10, w // 2)
        bh = rng.randint(h // 10, h // 2)
        # Semi-transparent overlay
        alpha = rng.uniform(0.3, 0.9)
        region = canvas[by:by + bh, bx:bx + bw].astype(np.float32)
        overlay = np.full_like(region, color, dtype=np.float32)
        canvas[by:by + bh, bx:bx + bw] = (region * (1 - alpha) + overlay * alpha).astype(np.uint8)
    # Add texture noise
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    noise = nrng.integers(-15, 16, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _bg_perlin_like(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Multi-octave smooth blobs with vivid colors."""
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    # Coarse blobs
    small = nrng.integers(30, 240, size=(rng.randint(3, 6), rng.randint(3, 6), 3), dtype=np.uint8)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # Add medium-frequency detail
    med = nrng.integers(0, 60, size=(rng.randint(10, 25), rng.randint(10, 25), 3), dtype=np.uint8)
    med_up = cv2.resize(med, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    canvas = canvas + med_up - 30
    # Fine grain
    noise = nrng.integers(-10, 11, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    return canvas


def _bg_diagonal_hatching(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Diagonal line hatching pattern."""
    base_color = [rng.randint(80, 220) for _ in range(3)]
    line_color = [max(0, c - rng.randint(30, 80)) for c in base_color]
    canvas = np.full((h, w, 3), base_color, dtype=np.uint8)
    spacing = rng.randint(6, 20)
    thickness = rng.randint(1, 3)
    for offset in range(-max(h, w), max(h, w), spacing):
        pt1 = (offset, 0)
        pt2 = (offset + h, h)
        cv2.line(canvas, pt1, pt2, line_color, thickness)
    # Second direction for cross-hatch (50% chance)
    if rng.random() < 0.5:
        line_color2 = [min(255, c + rng.randint(10, 50)) for c in base_color]
        for offset in range(-max(h, w), max(h, w), spacing):
            pt1 = (w - offset, 0)
            pt2 = (w - offset - h, h)
            cv2.line(canvas, pt1, pt2, line_color2, thickness)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    noise = nrng.integers(-8, 9, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _bg_circles(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Random overlapping circles/dots pattern."""
    base_color = [rng.randint(60, 200) for _ in range(3)]
    canvas = np.full((h, w, 3), base_color, dtype=np.uint8)
    n_circles = rng.randint(15, 60)
    for _ in range(n_circles):
        color = [rng.randint(20, 245) for _ in range(3)]
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        radius = rng.randint(max(5, min(h, w) // 30), max(10, min(h, w) // 5))
        thickness = rng.choice([-1, 2, 3])  # -1 = filled
        cv2.circle(canvas, (cx, cy), radius, color, thickness)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    noise = nrng.integers(-10, 11, size=(h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def random_background(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Generate a random textured background."""
    generators = [
        lambda: _bg_solid_noisy(h, w, rng),
        lambda: _bg_gradient_multi(h, w, rng),
        lambda: _bg_wood_like(h, w, rng),
        lambda: _bg_fabric_like(h, w, rng),
        lambda: _bg_concrete_like(h, w, rng),
        lambda: _bg_multi_color_blocks(h, w, rng),
        lambda: _bg_perlin_like(h, w, rng),
        lambda: _bg_diagonal_hatching(h, w, rng),
        lambda: _bg_circles(h, w, rng),
    ]
    return rng.choice(generators)()


# ---------------------------------------------------------------------------
# White background removal
# ---------------------------------------------------------------------------


def _remove_white_bg(img: np.ndarray, threshold: int = 235) -> tuple[np.ndarray, np.ndarray]:
    """Create an alpha mask via flood-fill from corners/edges.

    Returns (img_rgb, alpha_mask) where alpha is 0 for bg, 255 for product.
    """
    h, w = img.shape[:2]
    is_bright = (img[:, :, 0] > threshold) & (img[:, :, 1] > threshold) & (img[:, :, 2] > threshold)
    bright_u8 = is_bright.astype(np.uint8) * 255

    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seeds = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
    ]
    for px, py in seeds:
        if bright_u8[py, px] == 255:
            fill_mask[:] = 0
            cv2.floodFill(bright_u8, fill_mask, (px, py), 128)

    fg_mask = ((bright_u8 != 128) * 255).astype(np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0)
    return img, fg_mask


# ---------------------------------------------------------------------------
# Product image bank
# ---------------------------------------------------------------------------


class ProductImageBank:
    """Loads and indexes all product reference images by category."""

    def __init__(self, product_dir: Path, metadata_path: Path, coco_path: Path,
                 fp_crops_dir: Path | None = None):
        self.product_dir = product_dir

        with open(metadata_path) as f:
            meta = json.load(f)
        with open(coco_path) as f:
            coco = json.load(f)

        # Load false positive crops as hard negatives
        self.fp_crops: list[Path] = []
        if fp_crops_dir is not None and fp_crops_dir.exists():
            self.fp_crops = sorted(fp_crops_dir.glob("fp_*.jpg"))
            print(f"Loaded {len(self.fp_crops)} false positive crops as hard negatives")

        name_to_cat_id = {c["name"]: c["id"] for c in coco["categories"]}
        self.cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        self.all_cat_ids = list(name_to_cat_id.values())

        self.images_by_cat: dict[int, list[Path]] = defaultdict(list)
        self.n_loaded = 0

        for p in meta["products"]:
            name = p["product_name"]
            cat_id = name_to_cat_id.get(name)
            if cat_id is None or not p.get("has_images", False):
                continue
            code_dir = product_dir / p["product_code"]
            if not code_dir.exists():
                continue
            for view in ALL_VIEWS:
                img_path = code_dir / f"{view}.jpg"
                if img_path.exists():
                    self.images_by_cat[cat_id].append(img_path)
                    self.n_loaded += 1

        self.confusion_pairs_by_cat: dict[int, list[int]] = defaultdict(list)
        for a, b in CONFUSION_PAIRS:
            if a in self.images_by_cat and b in self.images_by_cat:
                self.confusion_pairs_by_cat[a].append(b)
                self.confusion_pairs_by_cat[b].append(a)

        ann_counts = Counter(a["category_id"] for a in coco["annotations"])
        self.ann_counts = ann_counts
        self.cats_with_images = sorted(
            self.images_by_cat.keys(),
            key=lambda c: ann_counts.get(c, 0),
        )

        # Pre-compute inverse-freq weights
        self._inv_weights = []
        for cat_id in self.cats_with_images:
            count = ann_counts.get(cat_id, 1)
            self._inv_weights.append(1.0 / max(1.0, count ** 0.5))
        total = sum(self._inv_weights)
        self._inv_weights = [w / total for w in self._inv_weights]

        print(f"ProductImageBank: {self.n_loaded} images across {len(self.images_by_cat)} categories")

    def sample_product(self, cat_id: int, rng: random.Random) -> np.ndarray | None:
        paths = self.images_by_cat.get(cat_id, [])
        if not paths:
            return None
        path = rng.choice(paths)
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None

    def sample_category_uniform(self, rng: random.Random) -> int:
        return rng.choice(self.cats_with_images)

    def sample_category_inverse_freq(self, rng: random.Random) -> int:
        return rng.choices(self.cats_with_images, weights=self._inv_weights, k=1)[0]

    def sample_fp_crop(self, rng: random.Random) -> np.ndarray | None:
        """Sample a random false positive crop (hard negative)."""
        if not self.fp_crops:
            return None
        path = rng.choice(self.fp_crops)
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_shelf_image(
    bank: ProductImageBank,
    rng: random.Random,
    canvas_w: int = 2000,
    canvas_h: int = 1500,
    n_products: tuple[int, int] = (60, 180),
    product_size_range: tuple[int, int] = (50, 250),
    p_hard_pair: float = 0.4,
    p_inverse_freq: float = 0.6,
) -> tuple[np.ndarray, list[dict]]:
    """Generate a single synthetic shelf image with annotations.

    Returns (image_rgb, annotations) where annotations are COCO-format
    dicts with bbox [x, y, w, h] and category_id.
    """
    # --- Random textured background ---
    canvas = random_background(canvas_h, canvas_w, rng)

    # Optionally add shelf lines on top of background
    if rng.random() < 0.5:
        n_lines = rng.randint(2, 7)
        line_ys = sorted(rng.sample(range(50, canvas_h - 30), min(n_lines, canvas_h - 80)))
        for ly in line_ys:
            color = (rng.randint(60, 180), rng.randint(60, 180), rng.randint(60, 180))
            thickness = rng.randint(1, 4)
            cv2.line(canvas, (0, ly), (canvas_w, ly), color, thickness)

    # --- Decide products ---
    n_prod = rng.randint(*n_products)
    categories_to_place = []
    for _ in range(n_prod):
        if rng.random() < p_hard_pair and categories_to_place:
            recent_cat = rng.choice(categories_to_place[-min(5, len(categories_to_place)):])
            counterparts = bank.confusion_pairs_by_cat.get(recent_cat, [])
            if counterparts:
                categories_to_place.append(rng.choice(counterparts))
                continue
        if rng.random() < p_inverse_freq:
            categories_to_place.append(bank.sample_category_inverse_freq(rng))
        else:
            categories_to_place.append(bank.sample_category_uniform(rng))

    # --- Row-based shelf layout with overlap ---
    n_rows = rng.randint(4, 10)
    row_height = canvas_h // n_rows
    # Allow rows to overlap slightly
    row_overlap = int(row_height * rng.uniform(0, 0.1))

    annotations = []
    placed_boxes = []
    placed_fp_boxes = []
    cat_idx = 0

    # Distribute products across rows, giving more to early rows
    products_per_row = [0] * n_rows
    for i in range(n_prod):
        products_per_row[i % n_rows] += 1

    for row_idx in range(n_rows):
        if cat_idx >= len(categories_to_place):
            break

        row_y_start = row_idx * row_height - row_overlap * row_idx
        row_y_end = row_y_start + row_height
        avail_h = row_y_end - row_y_start
        if avail_h < 20:
            cat_idx += products_per_row[row_idx]
            continue

        cursor_x = rng.randint(-5, 5)  # can start slightly off-canvas
        # Negative or zero gap = overlap between adjacent products
        gap = rng.randint(-8, 5)

        items_in_row = 0
        for _ in range(products_per_row[row_idx]):
            # Every few items, insert an FP crop as a hard negative
            items_in_row += 1
            if bank.fp_crops and items_in_row % rng.randint(3, 6) == 0:
                fp_img = bank.sample_fp_crop(rng)
                if fp_img is not None:
                    fp_h = int(avail_h * rng.uniform(0.5, 0.9))
                    fp_aspect = fp_img.shape[1] / max(1, fp_img.shape[0])
                    fp_w = max(15, int(fp_h * fp_aspect))
                    fp_w = min(fp_w, canvas_w // 4)
                    if cursor_x + fp_w <= canvas_w and fp_h >= 10 and fp_w >= 10:
                        fp_resized = cv2.resize(fp_img, (fp_w, fp_h), interpolation=cv2.INTER_LINEAR)
                        try:
                            fp_resized = PRODUCT_AUG(image=fp_resized)["image"]
                        except Exception:
                            pass
                        fp_x = max(0, cursor_x)
                        base_y = row_y_start + (avail_h - fp_h) // 2
                        jitter_y = rng.randint(-max(1, avail_h // 8), max(1, avail_h // 8))
                        fp_y = max(0, min(canvas_h - fp_h, base_y + jitter_y))
                        pw = min(fp_w, canvas_w - fp_x)
                        ph = min(fp_h, canvas_h - fp_y)
                        if pw >= 8 and ph >= 8:
                            canvas[fp_y:fp_y + ph, fp_x:fp_x + pw] = fp_resized[:ph, :pw]
                            placed_fp_boxes.append((fp_x, fp_y, fp_x + pw, fp_y + ph))
                            cursor_x += fp_w + gap
                            # No annotation — this is a true negative

            if cat_idx >= len(categories_to_place):
                break
            cat_id = categories_to_place[cat_idx]
            cat_idx += 1

            prod_img = bank.sample_product(cat_id, rng)
            if prod_img is None:
                continue

            prod_rgb, alpha_mask = _remove_white_bg(prod_img)

            # Target height — fill most of the row with variation
            target_h = int(avail_h * rng.uniform(0.55, 1.0))
            target_h = max(20, min(target_h, product_size_range[1]))
            aspect = prod_img.shape[1] / max(1, prod_img.shape[0])
            target_w = max(12, int(target_h * aspect))
            target_w = min(target_w, canvas_w // 3)

            if target_w < 10 or target_h < 10:
                continue
            # Allow products to extend slightly past right edge
            if cursor_x > canvas_w:
                continue

            # Resize
            prod_resized = cv2.resize(prod_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            alpha_resized = cv2.resize(alpha_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Apply per-product perspective + color augmentation
            try:
                aug_result = PRODUCT_AUG(image=prod_resized)
                prod_resized = aug_result["image"]
            except Exception:
                pass

            # Position: cursor_x with vertical jitter
            px = max(0, cursor_x)
            base_y = row_y_start + (avail_h - target_h) // 2
            jitter_y = rng.randint(-max(1, avail_h // 8), max(1, avail_h // 8))
            py = max(0, min(canvas_h - target_h, base_y + jitter_y))

            # Clip to canvas bounds
            paste_w = min(target_w, canvas_w - px)
            paste_h = min(target_h, canvas_h - py)
            if paste_w < 8 or paste_h < 8:
                cursor_x += target_w + gap
                continue

            # Alpha-blend
            alpha_3d = (alpha_resized[:paste_h, :paste_w, np.newaxis] / 255.0).astype(np.float32)
            region = canvas[py:py + paste_h, px:px + paste_w].astype(np.float32)
            blended = region * (1 - alpha_3d) + prod_resized[:paste_h, :paste_w].astype(np.float32) * alpha_3d
            canvas[py:py + paste_h, px:px + paste_w] = blended.astype(np.uint8)

            placed_boxes.append((px, py, px + paste_w, py + paste_h))
            annotations.append({
                "bbox": [int(px), int(py), int(paste_w), int(paste_h)],
                "category_id": int(cat_id),
            })
            cursor_x += target_w + gap

    # Whole-image augmentation
    try:
        canvas = IMAGE_AUG(image=canvas)["image"]
    except Exception:
        pass

    return canvas, annotations, placed_fp_boxes


def generate_synthetic_dataset(
    product_dir: Path,
    metadata_path: Path,
    coco_path: Path,
    output_dir: Path,
    n_images: int = 500,
    seed: int = 42,
    fp_crops_dir: Path | None = None,
):
    """Generate a full synthetic dataset with varied resolutions.

    Creates COCO-format JSON + images directory.
    """
    rng = random.Random(seed)
    bank = ProductImageBank(product_dir, metadata_path, coco_path, fp_crops_dir=fp_crops_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(exist_ok=True)

    with open(coco_path) as f:
        orig_coco = json.load(f)
    coco_categories = orig_coco["categories"]

    coco_images = []
    coco_annotations = []
    ann_id = 0
    cat_instance_count = Counter()

    # Resolution pool (varied sizes like real data)
    resolutions = [
        (880, 880),
        (1200, 900),
        (1500, 1200),
        (2000, 1500),
        (2500, 2000),
        (3000, 2000),
        (3000, 3000),
        (1280, 960),
        (900, 1200),   # portrait
        (1200, 1600),   # portrait
        (2000, 3000),   # portrait
        (4032, 3024),   # most common real resolution
        (3024, 4032),   # portrait version
    ]

    for img_idx in range(n_images):
        # Random resolution
        canvas_w, canvas_h = rng.choice(resolutions)

        # Scale product count with image area relative to 880x880
        # Real data: median 84 products per image at ~3000x3000
        area_ratio = (canvas_w * canvas_h) / (880 * 880)
        min_prods = max(20, int(40 * area_ratio))
        max_prods = max(min_prods + 15, int(100 * area_ratio))

        canvas, annotations, fp_boxes = generate_shelf_image(
            bank, rng,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            n_products=(min_prods, max_prods),
        )

        img_filename = f"synthetic_{img_idx:05d}.jpg"
        img_path = out_images_dir / img_filename
        Image.fromarray(canvas).save(img_path, quality=92)

        coco_images.append({
            "id": img_idx,
            "file_name": str(img_path.resolve()),
            "width": canvas_w,
            "height": canvas_h,
            "fp_boxes": [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in fp_boxes],
        })

        for ann in annotations:
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0,
            })
            cat_instance_count[ann["category_id"]] += 1
            ann_id += 1

        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            print(f"  [{img_idx + 1}/{n_images}] {canvas_w}x{canvas_h}, {len(annotations)} products, {ann_id} total annotations")

    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco_dict, f)

    print(f"\nDone: {n_images} images, {ann_id} annotations")
    print(f"Categories with instances: {len(cat_instance_count)}/{len(bank.images_by_cat)}")
    counts = sorted(cat_instance_count.values())
    print(f"Instances per category — min: {counts[0]}, max: {counts[-1]}, "
          f"median: {counts[len(counts)//2]}, mean: {sum(counts)/len(counts):.1f}")
    print(f"Output: {output_dir}")

    return coco_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/synthetic_shelves")
    parser.add_argument("--fp-crops-dir", type=str, default="data/false_positive_crops",
                        help="Dir with false positive crops to use as hard negatives")
    args = parser.parse_args()

    fp_dir = Path(args.fp_crops_dir) if args.fp_crops_dir else None

    generate_synthetic_dataset(
        product_dir=Path("data/product_images"),
        metadata_path=Path("data/product_images/metadata.json"),
        coco_path=Path("data/train/annotations.json"),
        output_dir=Path(args.output_dir),
        n_images=args.n_images,
        seed=args.seed,
        fp_crops_dir=fp_dir,
    )
