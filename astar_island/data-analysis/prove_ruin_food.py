"""
Prove: ruin reclaim food = nearest_food × 0.20

Ruin reclaims happen in phase 7 (last phase), so the observed food IS the
spawn food — no post-spawn modification.  We compare child food to the
nearest same-owner settlement's food in the SAME frame.

Reproduce: python3 data-analysis/prove_ruin_food.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUIN = 3


def load_all_replays():
    replays = []
    for rd in sorted(DATA_DIR.iterdir()):
        if not rd.is_dir():
            continue
        ad = rd / "analysis"
        if not ad.exists():
            continue
        for f in sorted(ad.glob("replay_seed_index=*.json")):
            with open(f) as fh:
                replays.append(json.load(fh))
    return replays


def main():
    replays = load_all_replays()

    child_food = []
    nearest_food = []

    for replay in replays:
        frames = replay["frames"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]

            before_positions = {
                (s["x"], s["y"]) for s in fb["settlements"] if s["alive"]
            }

            after_map = {}
            for s in fa["settlements"]:
                if s["alive"]:
                    after_map[(s["x"], s["y"])] = s

            for s in fa["settlements"]:
                if not s["alive"]:
                    continue
                pos = (s["x"], s["y"])
                if pos in before_positions:
                    continue
                x, y = pos
                if fb["grid"][y][x] != RUIN:
                    continue

                # Find nearest same-owner settlement in same frame
                owner = s["owner_id"]
                min_dist = float("inf")
                nf = None
                for apos, a_s in after_map.items():
                    if apos == pos or a_s["owner_id"] != owner:
                        continue
                    dx = apos[0] - x
                    dy = apos[1] - y
                    dist = dx * dx + dy * dy
                    if dist < min_dist:
                        min_dist = dist
                        nf = a_s["food"]

                if nf is not None and nf > 0.01:
                    child_food.append(s["food"])
                    nearest_food.append(nf)

    child_food = np.array(child_food)
    nearest_food = np.array(nearest_food)
    ratios = child_food / nearest_food
    n = len(ratios)

    print(f"Ruin reclaims with valid nearest: {n}")
    print(f"Median ratio: {np.median(ratios):.6f}")
    print(f"Mean  ratio:  {ratios.mean():.6f}")
    exact_020 = np.sum(np.abs(ratios - 0.20) < 0.001)
    print(f"Exact at 0.20 (±0.001): {exact_020} / {n} = {100 * exact_020 / n:.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Scatter: child food vs nearest food
    ax = axes[0]
    ax.scatter(nearest_food, child_food, s=1.5, alpha=0.08,
               color="darkorange", rasterized=True)
    lim = nearest_food.max() * 1.05
    ax.plot([0, lim], [0, lim * 0.20], color="red", lw=2,
            label="y = 0.20 x")
    ax.set_xlabel("nearest same-owner food (same frame)", fontsize=11)
    ax.set_ylabel("ruin reclaim food", fontsize=11)
    ax.set_title("Ruin reclaim food vs nearest food", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, child_food.max() * 1.05)

    # 2. Histogram of ratio
    ax = axes[1]
    ax.hist(ratios, bins=200, range=(0, 1.0), density=True,
            color="darkorange", alpha=0.7, edgecolor="none")
    ax.axvline(0.20, color="red", lw=2, label="0.20")
    ax.set_xlabel("child_food / nearest_food", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(f"Ratio distribution  (median = {np.median(ratios):.3f})",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.0)

    # 3. Residual (child - 0.20 × nearest)
    residual = child_food - 0.20 * nearest_food
    ax = axes[2]
    ax.hist(residual, bins=200, density=True,
            color="darkorange", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="red", lw=2, label="0")
    mae = np.mean(np.abs(residual))
    ax.set_xlabel("child_food − 0.20 × nearest_food", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(f"Residual  (MAE = {mae:.4f})", fontsize=12)
    ax.legend(fontsize=10)

    fig.suptitle(
        f"Ruin reclaim: food = nearest_food × 0.20   "
        f"(n = {n},  {100 * exact_020 / n:.0f}% exact)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig("prove_ruin_food.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved prove_ruin_food.png")


if __name__ == "__main__":
    main()
