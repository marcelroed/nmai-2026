#!/usr/bin/env python3
"""Deep classification error analysis with plots and confusion matrix.

Usage:
    uv run python classification_analysis.py --checkpoint checkpoint_1e-3_train_split.pth
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from PIL import Image

from norgesgruppen.config import MODEL_CLS, NUM_CLASSES
from norgesgruppen.patching import predict_with_patches
from norgesgruppen.postprocess import DEFAULT_TRANSFORMS, apply_transforms
from norgesgruppen.scoring import _ap_from_matches, _compute_iou_matrix, _match_predictions

# ---------------------------------------------------------------------------
HOLDOUT_IMAGES = [
    "img_00005.jpg", "img_00012.jpg", "img_00026.jpeg", "img_00036.jpg",
    "img_00062.jpg", "img_00075.jpg", "img_00088.jpg", "img_00105.jpg",
    "img_00108.jpg", "img_00122.jpg", "img_00152.jpeg", "img_00163.jpg",
    "img_00168.jpg", "img_00178.jpg", "img_00184.jpeg", "img_00193.jpg",
    "img_00243.jpg", "img_00261.jpg", "img_00271.jpg", "img_00296.jpg",
    "img_00302.jpg", "img_00304.jpg", "img_00317.jpg", "img_00320.jpg",
    "img_00373.jpg",
]
IMAGES_DIR = Path("data/train/images")
ANNOTATIONS_PATH = Path("data/train/annotations.json")


def load_gt(annotations_path: Path = ANNOTATIONS_PATH):
    with open(annotations_path) as f:
        data = json.load(f)
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    # Handle both bare filenames and paths like "data/train/images/img_00001.jpg"
    img_by_fname = {}
    for img in data["images"]:
        fname = Path(img["file_name"]).name
        img_by_fname[fname] = img
    anns_by_imgid = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_imgid[ann["image_id"]].append(ann)
    gt_by_fname = {}
    for fname, img_info in img_by_fname.items():
        anns = anns_by_imgid[img_info["id"]]
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        gt_by_fname[fname] = {
            "boxes": np.array(boxes).reshape(-1, 4),
            "labels": np.array(labels, dtype=int),
        }
    return cat_names, gt_by_fname


def load_product_metadata():
    """Load product metadata, return name->has_images mapping."""
    import subprocess, io
    result = subprocess.run(
        ["unzip", "-p", "NM_NGD_product_images.zip", "metadata.json"],
        capture_output=True,
    )
    if result.returncode != 0:
        return {}
    meta = json.loads(result.stdout)
    return {
        p["product_name"]: p["has_images"]
        for p in meta["products"]
    }


def load_model(checkpoint_path):
    import argparse as _argparse
    model = MODEL_CLS()
    model.model.reinitialize_detection_head(num_classes=NUM_CLASSES + 1)
    torch.serialization.add_safe_globals([_argparse.Namespace])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = "ema_model" if "ema_model" in ckpt else "model"
    model.model.model.load_state_dict(ckpt[state_key])
    print(f"Loaded {state_key} weights (epoch {ckpt.get('epoch', '?')})")
    model.model.model.eval().cuda()
    return model


def predict_single(model, image, threshold=0.2):
    dets = predict_with_patches(image, lambda img: model.predict(img, threshold=threshold))
    return apply_transforms(dets, DEFAULT_TRANSFORMS)


def run_predictions(model, gt_by_fname):
    """Run model on holdout, return (ground_truths, predictions) lists."""
    ground_truths, predictions = [], []
    for img_name in HOLDOUT_IMAGES:
        image_path = IMAGES_DIR / img_name
        if not image_path.exists() or img_name not in gt_by_fname:
            continue
        print(f"  {img_name}...", end=" ", flush=True)
        gt = gt_by_fname[img_name]
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            dets = predict_single(model, image)
        if dets.is_empty():
            pred = {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,), dtype=int), "scores": np.zeros((0,))}
        else:
            pred = {"boxes": dets.xyxy, "labels": dets.class_id, "scores": dets.confidence}
        ground_truths.append(gt)
        predictions.append(pred)
        print(f"GT={len(gt['boxes'])} Pred={len(pred['boxes'])}")
    return ground_truths, predictions


def compute_per_class_stats(ground_truths, predictions, cat_names):
    """Compute detailed per-class classification stats.

    Returns dict: cat_id -> {ap, n_gt, tp, fp, fn, confused_as, confused_from}
    """
    per_class = defaultdict(lambda: {
        "scores": [], "matched": [], "n_gt": 0,
        "confused_as": Counter(),    # this GT class was predicted as...
        "confused_from": Counter(),  # this predicted class was actually...
    })

    for gt, pred in zip(ground_truths, predictions):
        gt_boxes = gt["boxes"] if len(gt["boxes"]) > 0 else np.zeros((0, 4))
        gt_labels = gt["labels"] if len(gt["labels"]) > 0 else np.zeros((0,), dtype=int)
        pred_boxes = pred["boxes"] if len(pred["boxes"]) > 0 else np.zeros((0, 4))
        pred_scores = pred["scores"] if len(pred["scores"]) > 0 else np.zeros((0,))
        pred_labels = pred["labels"] if len(pred["labels"]) > 0 else np.zeros((0,), dtype=int)

        for lbl in gt_labels:
            per_class[int(lbl)]["n_gt"] += 1

        # Class-aware matching
        scores_cls, matched_cls = _match_predictions(
            gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
            iou_threshold=0.5, check_class=True,
        )
        for j in range(len(pred_labels)):
            lbl = int(pred_labels[j])
            per_class[lbl]["scores"].append(scores_cls[j])
            per_class[lbl]["matched"].append(matched_cls[j])

        # Class-agnostic matching to find confusion pairs
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        iou_matrix = _compute_iou_matrix(pred_boxes, gt_boxes)
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        for pred_idx in np.argsort(-pred_scores):
            ious = iou_matrix[pred_idx].copy()
            ious[gt_matched] = 0.0
            best_gt = np.argmax(ious)
            if ious[best_gt] >= 0.5:
                gt_matched[best_gt] = True
                gt_lbl = int(gt_labels[best_gt])
                pred_lbl = int(pred_labels[pred_idx])
                if gt_lbl != pred_lbl:
                    per_class[gt_lbl]["confused_as"][pred_lbl] += 1
                    per_class[pred_lbl]["confused_from"][gt_lbl] += 1

    # Compute AP per class
    results = {}
    for lbl, d in per_class.items():
        if d["n_gt"] == 0 and not d["scores"]:
            continue
        scores = np.array(d["scores"]) if d["scores"] else np.array([])
        matched = np.array(d["matched"], dtype=bool) if d["matched"] else np.array([], dtype=bool)
        ap = _ap_from_matches(scores, matched, d["n_gt"]) if d["n_gt"] > 0 else 0.0
        tp = int(matched.sum())
        fp = int((~matched).sum()) if len(matched) > 0 else 0
        fn = d["n_gt"] - tp
        results[lbl] = {
            "ap": ap, "n_gt": d["n_gt"], "tp": tp, "fp": fp, "fn": fn,
            "confused_as": d["confused_as"],
            "confused_from": d["confused_from"],
            "n_confused_as": sum(d["confused_as"].values()),
            "n_confused_from": sum(d["confused_from"].values()),
        }
    return results


def compute_score_impact(per_class, cat_names):
    """For each class, compute how much it drags down the overall cls mAP."""
    # cls_mAP = mean of per-class APs. Each class contributes 1/N.
    n_classes = sum(1 for d in per_class.values() if d["n_gt"] > 0)
    impacts = []
    for lbl, d in per_class.items():
        if d["n_gt"] == 0:
            continue
        # Score lost by this class = (1.0 - AP) / n_classes
        # Weighted by 0.3 for combined score impact
        cls_map_loss = (1.0 - d["ap"]) / n_classes
        combined_loss = 0.3 * cls_map_loss
        impacts.append({
            "cat_id": lbl,
            "name": cat_names.get(lbl, f"cat_{lbl}"),
            "ap": d["ap"],
            "n_gt": d["n_gt"],
            "tp": d["tp"],
            "fp": d["fp"],
            "fn": d["fn"],
            "n_confused_as": d["n_confused_as"],
            "top_confusion": d["confused_as"].most_common(1)[0] if d["confused_as"] else None,
            "cls_map_loss": cls_map_loss,
            "combined_loss": combined_loss,
        })
    impacts.sort(key=lambda x: -x["cls_map_loss"])
    return impacts


def build_confusion_matrix(per_class, cat_names, min_confusions=1):
    """Build confusion matrix for categories with significant confusion."""
    # Collect all confusion pairs
    pairs = Counter()
    for lbl, d in per_class.items():
        for pred_lbl, count in d["confused_as"].items():
            pairs[(lbl, pred_lbl)] += count

    # Find categories involved in confusion (only pairs with real misclassifications)
    involved = set()
    for (gt, pred), count in pairs.items():
        if count >= min_confusions:
            involved.add(gt)
            involved.add(pred)

    involved = sorted(involved)
    n = len(involved)
    idx_map = {lbl: i for i, lbl in enumerate(involved)}

    matrix = np.zeros((n, n))
    for (gt, pred), count in pairs.items():
        if gt in idx_map and pred in idx_map:
            matrix[idx_map[gt], idx_map[pred]] = count

    labels = [cat_names.get(lbl, str(lbl)) for lbl in involved]
    return matrix, labels, involved


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ap_distribution(per_class, cat_names, save_dir):
    """Bar chart of per-class AP, sorted, colored by AP range."""
    cats_with_gt = [(lbl, d) for lbl, d in per_class.items() if d["n_gt"] > 0]
    cats_with_gt.sort(key=lambda x: x[1]["ap"])

    aps = [d["ap"] for _, d in cats_with_gt]
    names = [cat_names.get(lbl, str(lbl)) for lbl, _ in cats_with_gt]

    fig, ax = plt.subplots(figsize=(14, max(8, len(aps) * 0.09)))
    colors = []
    for ap in aps:
        if ap == 0.0:
            colors.append("#d32f2f")
        elif ap < 0.5:
            colors.append("#f57c00")
        elif ap < 0.8:
            colors.append("#fbc02d")
        elif ap < 1.0:
            colors.append("#388e3c")
        else:
            colors.append("#1b5e20")

    y = np.arange(len(aps))
    ax.barh(y, aps, color=colors, height=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([n[:40] for n in names], fontsize=4.5)
    ax.set_xlabel("Classification AP@0.5")
    ax.set_title(f"Per-Class Classification AP (n={len(aps)} classes)\nRed=0, Orange=<0.5, Yellow=<0.8, Green=≥0.8, Dark green=1.0")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=np.mean(aps), color="blue", linestyle="--", linewidth=1, label=f"Mean={np.mean(aps):.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_dir / "per_class_ap.png", dpi=180)
    plt.close(fig)
    print(f"  Saved {save_dir / 'per_class_ap.png'}")


def plot_score_impact(impacts, save_dir, top_n=40):
    """Bar chart: combined score lost per category."""
    top = impacts[:top_n]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.25)))
    names = [f"{x['name'][:35]} (AP={x['ap']:.2f})" for x in top]
    losses = [x["combined_loss"] for x in top]
    y = np.arange(len(top))

    ax.barh(y, losses, color="#d32f2f", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Combined score lost (points)")
    ax.set_title(f"Top {top_n} Categories by Score Impact\n(each bar = how much this class drags down the combined score)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_dir / "score_impact.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_dir / 'score_impact.png'}")


def plot_confusion_matrix(matrix, labels, save_dir):
    """Heatmap of confusion between categories."""
    # Filter to only rows/cols with any confusion
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    active = (row_sums > 0) | (col_sums > 0)
    matrix = matrix[active][:, active]
    labels = [l for l, a in zip(labels, active) if a]

    n = len(labels)
    if n == 0:
        return

    # Truncate labels
    short_labels = [l[:30] for l in labels]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(8, n * 0.35)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=5.5, ha="right")
    ax.set_yticklabels(short_labels, fontsize=5.5)
    ax.set_xlabel("Predicted as", fontsize=10)
    ax.set_ylabel("Ground truth", fontsize=10)
    ax.set_title("Classification Confusion Matrix (IoU≥0.5, count of misclassifications)")

    # Annotate cells with counts
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        fontsize=5, color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_dir / 'confusion_matrix.png'}")


def plot_error_type_breakdown(per_class, cat_names, save_dir, top_n=40):
    """Stacked bar: TP/FN/misclassified-as/misclassified-from/FP for worst categories."""
    # Sort by most total errors
    items = []
    for lbl, d in per_class.items():
        if d["n_gt"] == 0:
            continue
        total_err = d["fn"] + d["fp"] + d["n_confused_as"] + d["n_confused_from"]
        if total_err > 0:
            items.append((lbl, d, total_err))
    items.sort(key=lambda x: -x[2])
    items = items[:top_n]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.25)))
    y = np.arange(len(items))
    names = [cat_names.get(lbl, str(lbl))[:35] for lbl, _, _ in items]

    tp = [d["tp"] for _, d, _ in items]
    fn = [d["fn"] for _, d, _ in items]
    confused_as = [d["n_confused_as"] for _, d, _ in items]
    fp = [d["fp"] for _, d, _ in items]

    ax.barh(y, tp, color="#388e3c", label="TP (correct)", height=0.7)
    left = np.array(tp, dtype=float)
    ax.barh(y, fn, left=left, color="#1565c0", label="FN (missed)", height=0.7)
    left += np.array(fn, dtype=float)
    ax.barh(y, confused_as, left=left, color="#f57c00", label="Misclassified (GT→wrong pred)", height=0.7)
    left += np.array(confused_as, dtype=float)
    ax.barh(y, fp, left=left, color="#d32f2f", label="FP (hallucinated)", height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("Count")
    ax.set_title(f"Error Type Breakdown — Top {top_n} Categories by Total Errors")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_dir / "error_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_dir / 'error_breakdown.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="classification_analysis")
    parser.add_argument("--annotations", type=str, default=str(ANNOTATIONS_PATH),
                        help="Path to COCO annotations JSON (default: data/train/annotations.json)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cat_names, gt_by_fname = load_gt(Path(args.annotations))
    product_imgs = load_product_metadata()
    model = load_model(args.checkpoint)

    print("Running predictions...")
    ground_truths, predictions = run_predictions(model, gt_by_fname)

    print("\nComputing per-class stats...")
    per_class = compute_per_class_stats(ground_truths, predictions, cat_names)

    print("Computing score impact...")
    impacts = compute_score_impact(per_class, cat_names)

    # =====================================================================
    # Plots
    # =====================================================================
    print("\nGenerating plots...")
    plot_ap_distribution(per_class, cat_names, save_dir)
    plot_score_impact(impacts, save_dir)

    matrix, labels, involved = build_confusion_matrix(per_class, cat_names)
    plot_confusion_matrix(matrix, labels, save_dir)

    plot_error_type_breakdown(per_class, cat_names, save_dir)

    # =====================================================================
    # Comprehensive table: worst products with score impact + ref images
    # =====================================================================
    print(f"\n{'='*120}")
    print(f"WORST PRODUCTS — CLASSIFICATION SCORE IMPACT TABLE")
    print(f"{'='*120}")
    header = f"{'Rank':>4} {'CombLoss':>9} {'AP':>5} {'GT':>3} {'TP':>3} {'FN':>3} {'FP':>3} {'Conf':>4} {'RefImg':>6}  {'Name':<38} {'Top confusion (→ predicted as)'}"
    print(header)
    print("-" * 120)

    cumulative_loss = 0.0
    for i, imp in enumerate(impacts[:60]):
        cat_id = imp["cat_id"]
        name = imp["name"]
        has_img = product_imgs.get(name, None)
        if has_img is True:
            img_str = "YES"
        elif has_img is False:
            img_str = "NO*"  # in metadata but no images
        else:
            img_str = "NO"   # not in metadata at all

        confusion_str = ""
        if imp["top_confusion"]:
            conf_lbl, conf_count = imp["top_confusion"]
            conf_name = cat_names.get(conf_lbl, str(conf_lbl))[:30]
            confusion_str = f"{conf_count}x → {conf_name}"

        cumulative_loss += imp["combined_loss"]
        print(f"{i+1:>4} {imp['combined_loss']:>8.5f} {imp['ap']:>5.2f} {imp['n_gt']:>3} {imp['tp']:>3} {imp['fn']:>3} {imp['fp']:>3} {imp['n_confused_as']:>4} {img_str:>6}  {name[:38]:<38} {confusion_str}")

    print(f"\n  Cumulative combined score loss from top 60: {cumulative_loss:.4f}")
    print(f"  Total classification gap:                   {0.3 * (1.0 - np.mean([d['ap'] for d in per_class.values() if d['n_gt'] > 0])):.4f}")

    # =====================================================================
    # Summary of products without reference images that matter
    # =====================================================================
    print(f"\n{'='*120}")
    print(f"CATEGORIES WITHOUT REFERENCE IMAGES — SORTED BY SCORE IMPACT")
    print(f"{'='*120}")
    no_img_impacts = [imp for imp in impacts if product_imgs.get(imp["name"]) is not True]
    print(f"{'Rank':>4} {'CombLoss':>9} {'AP':>5} {'GT':>3} {'TP':>3} {'FN':>3} {'Conf':>4}  Name")
    print("-" * 90)
    cumul = 0.0
    for i, imp in enumerate(no_img_impacts[:30]):
        cumul += imp["combined_loss"]
        print(f"{i+1:>4} {imp['combined_loss']:>8.5f} {imp['ap']:>5.2f} {imp['n_gt']:>3} {imp['tp']:>3} {imp['fn']:>3} {imp['n_confused_as']:>4}  {imp['name'][:50]}")
    print(f"\n  Cumulative loss from categories without ref images: {cumul:.4f}")

    # =====================================================================
    # Confusion clusters: groups of products that confuse each other
    # =====================================================================
    print(f"\n{'='*120}")
    print(f"CONFUSION CLUSTERS — groups of mutually confused products")
    print(f"{'='*120}")

    # Build adjacency from confusion pairs
    from collections import deque
    adj = defaultdict(set)
    for lbl, d in per_class.items():
        for pred_lbl, count in d["confused_as"].items():
            if count >= 2:
                adj[lbl].add(pred_lbl)
                adj[pred_lbl].add(lbl)

    # BFS to find connected components
    visited = set()
    clusters = []
    for node in adj:
        if node in visited:
            continue
        cluster = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            cluster.add(n)
            for neighbor in adj[n]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(cluster) >= 2:
            clusters.append(cluster)

    # Sort clusters by total score impact
    cluster_impacts = []
    for cluster in clusters:
        total_loss = sum(
            (1.0 - per_class[lbl]["ap"]) / sum(1 for d in per_class.values() if d["n_gt"] > 0) * 0.3
            for lbl in cluster if lbl in per_class and per_class[lbl]["n_gt"] > 0
        )
        cluster_impacts.append((cluster, total_loss))
    cluster_impacts.sort(key=lambda x: -x[1])

    for ci, (cluster, loss) in enumerate(cluster_impacts):
        members = sorted(cluster, key=lambda l: per_class.get(l, {}).get("ap", 1.0))
        print(f"\n  Cluster {ci+1} (combined score impact: {loss:.4f}):")
        for lbl in members:
            d = per_class.get(lbl, {})
            ap = d.get("ap", -1)
            n_gt = d.get("n_gt", 0)
            has_img = product_imgs.get(cat_names.get(lbl, ""), None)
            img_str = "YES" if has_img is True else "NO"
            confused = d.get("confused_as", Counter())
            conf_str = ", ".join(f"{cat_names.get(k,'?')[:20]}({v})" for k, v in confused.most_common(3)) if confused else ""
            print(f"    [{lbl:>3}] AP={ap:.2f} GT={n_gt:>2} img={img_str:<3}  {cat_names.get(lbl,'?')[:40]}")
            if conf_str:
                print(f"           confused as: {conf_str}")

    print(f"\nAll plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
