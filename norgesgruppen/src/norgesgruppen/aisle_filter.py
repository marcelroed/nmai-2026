"""Aisle-based filtering for inference predictions.

Uses co-occurrence statistics from training data to suppress predictions
of categories that are unlikely to appear together in the same image.

Usage:
    groups = build_aisle_groups("data/train/annotations.json")
    filter_fn = aisle_filter(groups, min_group_fraction=0.05)
    detections = apply_transforms(detections, [filter_fn])
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv

from norgesgruppen.postprocess import Transform


def build_cooccurrence(annotations_path: str | Path) -> tuple[np.ndarray, list[int]]:
    """Build the co-occurrence matrix from training annotations.

    Returns:
        cooccur: (N, N) matrix where cooccur[i,j] = number of images where
            category i and j both appear.
        cat_ids: sorted list of category IDs (index into the matrix).
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    cat_ids = sorted(c["id"] for c in coco["categories"])
    cat_idx = {c: i for i, c in enumerate(cat_ids)}
    n = len(cat_ids)

    img_cats = defaultdict(set)
    for ann in coco["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])

    cooccur = np.zeros((n, n), dtype=int)
    for cats in img_cats.values():
        cats = list(cats)
        for i in range(len(cats)):
            for j in range(len(cats)):
                cooccur[cat_idx[cats[i]], cat_idx[cats[j]]] += 1

    return cooccur, cat_ids


def build_aisle_groups(
    annotations_path: str | Path,
    min_cooccur: int = 1,
) -> list[set[int]]:
    """Build aisle groups via connected components on the co-occurrence graph.

    Two categories are in the same group if they co-occur in at least
    `min_cooccur` images.

    Args:
        annotations_path: Path to COCO annotations JSON.
        min_cooccur: Minimum co-occurrence count to consider two categories
            as being in the same aisle.

    Returns:
        List of sets, each containing category IDs that co-occur.
    """
    cooccur, cat_ids = build_cooccurrence(annotations_path)
    n = len(cat_ids)

    # Build adjacency from thresholded co-occurrence
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if cooccur[i, j] >= min_cooccur:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components
    visited = set()
    groups = []
    for i in range(n):
        if i in visited:
            continue
        group = set()
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.add(cat_ids[node])
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        groups.append(group)

    return groups


def identify_aisle(
    class_ids: np.ndarray,
    confidences: np.ndarray,
    groups: list[set[int]],
) -> set[int] | None:
    """Given predictions for one image, identify the most likely aisle group.

    Picks the group with the highest total confidence among predictions.
    Returns the set of allowed category IDs, or None if no group matches.
    """
    if len(class_ids) == 0:
        return None

    best_group = None
    best_score = -1.0

    for group in groups:
        mask = np.array([int(c) in group for c in class_ids])
        score = confidences[mask].sum() if mask.any() else 0.0
        if score > best_score:
            best_score = score
            best_group = group

    return best_group


def aisle_filter(
    groups: list[set[int]],
    min_group_fraction: float = 0.05,
) -> Transform:
    """Create a post-processing transform that filters out-of-aisle predictions.

    For each image's predictions:
    1. Identify the dominant aisle group (by total confidence).
    2. Remove predictions whose category is not in that group,
       UNLESS the prediction's confidence is very high (top min_group_fraction
       of all predictions), in which case keep it regardless.

    Args:
        groups: Aisle groups from build_aisle_groups().
        min_group_fraction: Keep out-of-group predictions if their confidence
            is in the top this fraction (safety valve for novel co-occurrences).
    """
    def _apply(detections: sv.Detections) -> sv.Detections:
        if detections.is_empty() or len(groups) <= 1:
            return detections

        allowed = identify_aisle(detections.class_id, detections.confidence, groups)
        if allowed is None:
            return detections

        in_group = np.array([int(c) in allowed for c in detections.class_id])

        # Safety valve: keep high-confidence out-of-group predictions
        if min_group_fraction > 0 and not in_group.all():
            conf_threshold = np.quantile(detections.confidence, 1.0 - min_group_fraction)
            high_conf = detections.confidence >= conf_threshold
            keep = in_group | high_conf
        else:
            keep = in_group

        n_filtered = (~keep).sum()
        if n_filtered > 0:
            print(f"  [AISLE FILTER] removed {n_filtered}/{len(detections)} out-of-aisle predictions")

        return detections[keep]

    return _apply


def cooccurrence_filter(
    cooccur: np.ndarray,
    cat_ids: list[int],
    min_support: int = 2,
    confidence_floor: float = 0.5,
) -> Transform:
    """Soft filter using co-occurrence statistics directly.

    For each prediction, check how many of the image's other (confident)
    predictions co-occur with it in the training data. If a category has
    zero co-occurrence with ALL other confident predictions in the image,
    suppress it.

    This is softer than hard aisle groups — it only removes predictions
    that are truly isolated from everything else in the image.

    Args:
        cooccur: Co-occurrence matrix from build_cooccurrence().
        cat_ids: Sorted list of category IDs (index into cooccur).
        min_support: Minimum number of confident "neighbor" predictions that
            must co-occur with a category to keep it. Default 2.
        confidence_floor: Only consider predictions above this confidence
            as "context" for deciding what belongs in the image.
    """
    cat_idx = {c: i for i, c in enumerate(cat_ids)}

    def _apply(detections: sv.Detections) -> sv.Detections:
        if detections.is_empty() or len(detections) < 3:
            return detections

        confident = detections.confidence >= confidence_floor
        context_cats = set(int(c) for c, conf in zip(detections.class_id, confident) if conf)

        if len(context_cats) < 2:
            return detections

        keep = np.ones(len(detections), dtype=bool)

        for i in range(len(detections)):
            cat = int(detections.class_id[i])
            if cat not in cat_idx:
                continue
            ci = cat_idx[cat]

            # Count how many context categories this one co-occurs with
            n_cooccur = 0
            for other_cat in context_cats:
                if other_cat == cat:
                    continue
                if other_cat not in cat_idx:
                    continue
                if cooccur[ci, cat_idx[other_cat]] > 0:
                    n_cooccur += 1

            if n_cooccur < min_support:
                keep[i] = False

        n_filtered = (~keep).sum()
        if n_filtered > 0:
            print(f"  [COOCCUR FILTER] removed {n_filtered}/{len(detections)} isolated predictions")

        return detections[keep]

    return _apply
