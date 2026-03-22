"""
Prove: growth spawn wealth = parent_wealth × 0.20

Growth spawns happen in phase 3.  Wealth is NOT modified by phases 1-2
(food production, population), so parent_wealth at spawn time equals
parent_wealth in the previous frame.

We identify the parent as the nearest same-owner settlement whose
population decreased between frames (i.e. it paid parent_cost).

Reproduce: python3 data-analysis/prove_growth_wealth.py
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

    child_wealth = []
    parent_wealth = []

    for replay in replays:
        frames = replay["frames"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]

            before_map = {}
            for s in fb["settlements"]:
                if s["alive"]:
                    before_map[(s["x"], s["y"])] = s

            after_map = {}
            for s in fa["settlements"]:
                if s["alive"]:
                    after_map[(s["x"], s["y"])] = s

            for s in fa["settlements"]:
                if not s["alive"]:
                    continue
                pos = (s["x"], s["y"])
                if pos in before_map:
                    continue
                x, y = pos
                # Growth spawns only (non-ruin)
                if fb["grid"][y][x] == RUIN:
                    continue

                owner = s["owner_id"]

                # Find parent: nearest same-owner that lost population
                min_dist = float("inf")
                best_parent = None
                for bpos, bs in before_map.items():
                    if bs["owner_id"] != owner:
                        continue
                    if bpos not in after_map:
                        continue
                    if after_map[bpos]["population"] >= bs["population"]:
                        continue
                    dx = bpos[0] - x
                    dy = bpos[1] - y
                    dist = dx * dx + dy * dy
                    if dist < min_dist:
                        min_dist = dist
                        best_parent = bs

                if best_parent is not None and best_parent["wealth"] > 0.005:
                    child_wealth.append(s["wealth"])
                    parent_wealth.append(best_parent["wealth"])

    child_wealth = np.array(child_wealth)
    parent_wealth = np.array(parent_wealth)
    ratios = child_wealth / parent_wealth
    n = len(ratios)

    print(f"Growth spawns with identified parent (wealth > 0.005): {n}")
    print(f"Median ratio: {np.median(ratios):.6f}")
    print(f"Mean  ratio:  {ratios.mean():.6f}")
    exact_020 = np.sum(np.abs(ratios - 0.20) < 0.001)
    print(f"Exact at 0.20 (±0.001): {exact_020} / {n} = {100 * exact_020 / n:.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Scatter: child wealth vs parent wealth
    ax = axes[0]
    ax.scatter(parent_wealth, child_wealth, s=1.5, alpha=0.08,
               color="steelblue", rasterized=True)
    lim = parent_wealth.max() * 1.05
    ax.plot([0, lim], [0, lim * 0.20], color="red", lw=2,
            label="y = 0.20 x")
    ax.set_xlabel("parent wealth (prev frame)", fontsize=11)
    ax.set_ylabel("child wealth at spawn", fontsize=11)
    ax.set_title("Growth: child wealth vs parent wealth", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, child_wealth.max() * 1.05)

    # 2. Histogram of ratio
    ax = axes[1]
    ax.hist(ratios, bins=200, range=(0, 1.0), density=True,
            color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0.20, color="red", lw=2, label="0.20")
    ax.set_xlabel("child_wealth / parent_wealth", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(f"Ratio distribution  (median = {np.median(ratios):.3f})",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.0)

    # 3. Residual (child - 0.20 × parent)
    residual = child_wealth - 0.20 * parent_wealth
    ax = axes[2]
    ax.hist(residual, bins=200, density=True,
            color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="red", lw=2, label="0")
    mae = np.mean(np.abs(residual))
    ax.set_xlabel("child_wealth − 0.20 × parent_wealth", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(f"Residual  (MAE = {mae:.4f})", fontsize=12)
    ax.legend(fontsize=10)

    fig.suptitle(
        f"Growth spawn: wealth = parent_wealth × 0.20   "
        f"(n = {n},  {100 * exact_020 / n:.0f}% exact)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig("prove_growth_wealth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved prove_growth_wealth.png")


if __name__ == "__main__":
    main()
