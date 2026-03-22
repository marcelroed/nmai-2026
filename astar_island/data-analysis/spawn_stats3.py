"""
Refined spawn stat analysis.

Key insight: observed ratios are noisy because:
- Growth spawns (phase 3): child goes through phases 4-6 after creation
- Ruin reclaims (phase 7): NO modification after creation → cleanest signal

Strategy:
1. For ruin reclaims: compare child_food to nearest food in SAME frame (i+1)
   since phase 7 is last and nearest food at phase 7 ≈ nearest food at frame i+1
2. For growth spawns: compare child food to parent food in SAME frame (i+1)
   Both see same winter, but parent also went through phases 4-6
3. Check wealth ratios for both

Reproduce: python3 data-analysis/spawn_stats3.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
                data = json.load(fh)
            replays.append(data)
    return replays


def main():
    replays = load_all_replays()

    RUIN = 3

    # ── Growth spawn analysis ────────────────────────────────────────────
    # Identify parent by: same owner, pop decreased, in previous frame
    growth_food_ratios_prev = []   # child_food / parent_food_prev_frame
    growth_food_ratios_same = []   # child_food / parent_food_same_frame
    growth_wealth_ratios_same = [] # child_wealth / parent_wealth_same_frame
    growth_child_food = []
    growth_parent_food_prev = []
    growth_parent_food_same = []
    growth_child_wealth = []
    growth_parent_wealth_prev = []
    growth_parent_wealth_same = []

    # ── Ruin reclaim analysis ────────────────────────────────────────────
    ruin_food_ratios_prev = []     # child_food / nearest_food_prev_frame
    ruin_food_ratios_same = []     # child_food / nearest_food_same_frame
    ruin_child_food = []
    ruin_nearest_food_same = []
    ruin_child_wealth = []

    for replay in replays:
        frames = replay["frames"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]

            # Build maps for frame before and after
            before_map = {}
            for s in fb["settlements"]:
                if s["alive"]:
                    before_map[(s["x"], s["y"])] = s

            after_map = {}
            for s in fa["settlements"]:
                if s["alive"]:
                    after_map[(s["x"], s["y"])] = s

            # Find new settlements in frame after
            for s in fa["settlements"]:
                if not s["alive"]:
                    continue
                pos = (s["x"], s["y"])
                if pos in before_map:
                    continue

                x, y = pos
                was_ruin = fb["grid"][y][x] == RUIN
                owner = s["owner_id"]

                if not was_ruin:
                    # ── GROWTH SPAWN ──
                    # Find parent: same owner, existed before, pop decreased
                    best_parent_prev = None
                    best_parent_same = None
                    min_dist = float("inf")

                    for bpos, bs in before_map.items():
                        if bs["owner_id"] != owner:
                            continue
                        bx, by = bpos
                        dx, dy = bx - x, by - y
                        dist = (dx*dx + dy*dy) ** 0.5

                        # Check if this settlement still exists in after frame
                        if bpos not in after_map:
                            continue
                        a_s = after_map[bpos]

                        # Parent should have lost population
                        if a_s["population"] >= bs["population"]:
                            continue

                        if dist < min_dist:
                            min_dist = dist
                            best_parent_prev = bs
                            best_parent_same = a_s

                    if best_parent_prev is not None and best_parent_prev["food"] > 0.05:
                        growth_child_food.append(s["food"])
                        growth_parent_food_prev.append(best_parent_prev["food"])
                        growth_parent_food_same.append(best_parent_same["food"])
                        growth_food_ratios_prev.append(s["food"] / best_parent_prev["food"])
                        growth_food_ratios_same.append(s["food"] / best_parent_same["food"])

                        growth_child_wealth.append(s["wealth"])
                        growth_parent_wealth_prev.append(best_parent_prev["wealth"])
                        growth_parent_wealth_same.append(best_parent_same["wealth"])
                        if best_parent_same["wealth"] > 0.005:
                            growth_wealth_ratios_same.append(s["wealth"] / best_parent_same["wealth"])

                else:
                    # ── RUIN RECLAIM ──
                    # Find nearest same-owner settlement in SAME frame (after)
                    nearest_food_same = None
                    nearest_food_prev = None
                    min_dist = float("inf")
                    for apos, a_s in after_map.items():
                        if apos == pos:
                            continue
                        if a_s["owner_id"] != owner:
                            continue
                        ax, ay = apos
                        dx, dy = ax - x, ay - y
                        dist = (dx*dx + dy*dy) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest_food_same = a_s["food"]
                            # Also find this settlement in previous frame
                            if apos in before_map:
                                nearest_food_prev = before_map[apos]["food"]

                    if nearest_food_same is not None and nearest_food_same > 0.05:
                        ruin_child_food.append(s["food"])
                        ruin_nearest_food_same.append(nearest_food_same)
                        ruin_food_ratios_same.append(s["food"] / nearest_food_same)
                        if nearest_food_prev is not None and nearest_food_prev > 0.05:
                            ruin_food_ratios_prev.append(s["food"] / nearest_food_prev)
                        ruin_child_wealth.append(s["wealth"])

    # Convert to arrays
    growth_food_ratios_prev = np.array(growth_food_ratios_prev)
    growth_food_ratios_same = np.array(growth_food_ratios_same)
    growth_wealth_ratios_same = np.array(growth_wealth_ratios_same)
    growth_child_food = np.array(growth_child_food)
    growth_parent_food_prev = np.array(growth_parent_food_prev)
    growth_parent_food_same = np.array(growth_parent_food_same)
    growth_child_wealth = np.array(growth_child_wealth)
    growth_parent_wealth_prev = np.array(growth_parent_wealth_prev)
    growth_parent_wealth_same = np.array(growth_parent_wealth_same)

    ruin_food_ratios_prev = np.array(ruin_food_ratios_prev)
    ruin_food_ratios_same = np.array(ruin_food_ratios_same)
    ruin_child_food = np.array(ruin_child_food)
    ruin_nearest_food_same = np.array(ruin_nearest_food_same)
    ruin_child_wealth = np.array(ruin_child_wealth)

    # ═══════════════════════════════════════════════════════════════════
    # RUIN RECLAIM — food (cleanest signal, phase 7 = last)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("RUIN RECLAIM — food (phase 7, no post-spawn modification)")
    print("=" * 70)
    print(f"\n  N = {len(ruin_food_ratios_same)}")

    print(f"\n  child_food / nearest_food_SAME_FRAME:")
    print(f"    mean={ruin_food_ratios_same.mean():.6f}  median={np.median(ruin_food_ratios_same):.6f}")
    print(f"    std={ruin_food_ratios_same.std():.6f}")
    print(f"    P05={np.percentile(ruin_food_ratios_same, 5):.6f}  P95={np.percentile(ruin_food_ratios_same, 95):.6f}")

    for r in [0.10, 0.15, 0.20, 0.25, 0.30]:
        exact = np.mean(np.abs(ruin_food_ratios_same - r) < 0.001) * 100
        mae = np.mean(np.abs(ruin_food_ratios_same - r))
        print(f"    ratio={r:.2f}: MAE={mae:.6f}  exact(±0.001)={exact:.1f}%")

    if len(ruin_food_ratios_prev) > 0:
        print(f"\n  child_food / nearest_food_PREV_FRAME:")
        print(f"    mean={ruin_food_ratios_prev.mean():.6f}  median={np.median(ruin_food_ratios_prev):.6f}")
        for r in [0.10, 0.15, 0.20, 0.25, 0.30]:
            exact = np.mean(np.abs(ruin_food_ratios_prev - r) < 0.001) * 100
            print(f"    ratio={r:.2f}: exact(±0.001)={exact:.1f}%")

    # Check if nearest in same frame is correct proxy
    # The actual nearest might be by distance, not first in list
    # Let's plot child_food vs nearest_food_same
    print(f"\n  Ruin wealth: mean={ruin_child_wealth.mean():.6f}, "
          f"zero count={np.sum(ruin_child_wealth == 0)} ({100*np.mean(ruin_child_wealth == 0):.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # GROWTH SPAWN — food
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("GROWTH SPAWN — food")
    print("=" * 70)
    print(f"\n  N = {len(growth_food_ratios_same)}")

    print(f"\n  child_food / parent_food_PREV_FRAME:")
    print(f"    mean={growth_food_ratios_prev.mean():.6f}  median={np.median(growth_food_ratios_prev):.6f}")
    for r in [0.10, 0.15, 0.20, 0.25, 0.30]:
        exact = np.mean(np.abs(growth_food_ratios_prev - r) < 0.001) * 100
        mae = np.mean(np.abs(growth_food_ratios_prev - r))
        print(f"    ratio={r:.2f}: MAE={mae:.6f}  exact(±0.001)={exact:.1f}%")

    print(f"\n  child_food / parent_food_SAME_FRAME:")
    print(f"    mean={growth_food_ratios_same.mean():.6f}  median={np.median(growth_food_ratios_same):.6f}")
    print(f"    std={growth_food_ratios_same.std():.6f}")
    for r in [0.10, 0.15, 0.20, 0.25, 0.30]:
        exact = np.mean(np.abs(growth_food_ratios_same - r) < 0.001) * 100
        mae = np.mean(np.abs(growth_food_ratios_same - r))
        print(f"    ratio={r:.2f}: MAE={mae:.6f}  exact(±0.001)={exact:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # GROWTH SPAWN — wealth
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("GROWTH SPAWN — wealth")
    print("=" * 70)

    print(f"\n  child_wealth / parent_wealth_SAME_FRAME (parent > 0.005):")
    print(f"    N = {len(growth_wealth_ratios_same)}")
    print(f"    mean={growth_wealth_ratios_same.mean():.6f}  median={np.median(growth_wealth_ratios_same):.6f}")
    for r in [0.05, 0.10, 0.15, 0.20, 0.25]:
        exact = np.mean(np.abs(growth_wealth_ratios_same - r) < 0.001) * 100
        mae = np.mean(np.abs(growth_wealth_ratios_same - r))
        print(f"    ratio={r:.2f}: MAE={mae:.6f}  exact(±0.001)={exact:.1f}%")

    # Also check: is child_wealth = parent_wealth_prev * 0.20?
    has_wealth = growth_parent_wealth_prev > 0.005
    if has_wealth.sum() > 0:
        ratios_prev = growth_child_wealth[has_wealth] / growth_parent_wealth_prev[has_wealth]
        print(f"\n  child_wealth / parent_wealth_PREV_FRAME (parent > 0.005):")
        print(f"    N = {has_wealth.sum()}")
        print(f"    mean={ratios_prev.mean():.6f}  median={np.median(ratios_prev):.6f}")
        for r in [0.10, 0.15, 0.20, 0.25]:
            exact = np.mean(np.abs(ratios_prev - r) < 0.001) * 100
            print(f"    ratio={r:.2f}: exact(±0.001)={exact:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SCATTER PLOTS
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Ruin: child food vs nearest food (same frame)
    ax = axes[0, 0]
    ax.scatter(ruin_nearest_food_same, ruin_child_food, s=2, alpha=0.15,
               color="orange", rasterized=True)
    lim = ruin_nearest_food_same.max() * 1.05
    ax.plot([0, lim], [0, lim*0.20], "r-", lw=2, label="y = 0.20x")
    ax.plot([0, lim], [0, lim*0.15], "g--", lw=1.5, label="y = 0.15x")
    ax.plot([0, lim], [0, lim*0.25], "b--", lw=1.5, label="y = 0.25x")
    ax.set_xlabel("nearest food (same frame)")
    ax.set_ylabel("child food")
    ax.set_title("Ruin reclaim: food vs nearest (same frame)")
    ax.legend(fontsize=8)

    # 2. Ruin: food ratio histogram
    ax = axes[0, 1]
    ax.hist(ruin_food_ratios_same, bins=100, density=True, color="orange", alpha=0.7)
    ax.axvline(0.20, color="r", lw=2, label="0.20")
    ax.set_xlabel("child_food / nearest_food (same frame)")
    ax.set_title("Ruin reclaim: food ratio distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.0)

    # 3. Growth: child food vs parent food (same frame)
    ax = axes[0, 2]
    ax.scatter(growth_parent_food_same, growth_child_food, s=2, alpha=0.15,
               color="steelblue", rasterized=True)
    lim = growth_parent_food_same.max() * 1.05
    ax.plot([0, lim], [0, lim*0.10], "k--", lw=1.5, label="y = 0.10x")
    ax.plot([0, lim], [0, lim*0.20], "r-", lw=2, label="y = 0.20x")
    ax.plot([0, lim], [0, lim*0.25], "b--", lw=1.5, label="y = 0.25x")
    ax.set_xlabel("parent food (same frame)")
    ax.set_ylabel("child food")
    ax.set_title("Growth: food vs parent (same frame)")
    ax.legend(fontsize=8)

    # 4. Growth: food ratio histogram (same frame)
    ax = axes[1, 0]
    ax.hist(growth_food_ratios_same, bins=100, density=True, color="steelblue", alpha=0.7)
    ax.axvline(0.20, color="r", lw=2, label="0.20")
    ax.axvline(0.10, color="k", lw=1.5, ls="--", label="0.10")
    ax.set_xlabel("child_food / parent_food (same frame)")
    ax.set_title("Growth: food ratio distribution (same frame)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.0)

    # 5. Growth: food ratio histogram (prev frame)
    ax = axes[1, 1]
    ax.hist(growth_food_ratios_prev, bins=100, density=True, color="steelblue", alpha=0.7)
    ax.axvline(0.20, color="r", lw=2, label="0.20")
    ax.axvline(0.25, color="b", lw=1.5, ls="--", label="0.25")
    ax.set_xlabel("child_food / parent_food (prev frame)")
    ax.set_title("Growth: food ratio distribution (prev frame)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.0)

    # 6. Growth: wealth ratio histogram
    ax = axes[1, 2]
    ax.hist(growth_wealth_ratios_same, bins=100, density=True, color="steelblue", alpha=0.7)
    ax.axvline(0.20, color="r", lw=2, label="0.20")
    ax.set_xlabel("child_wealth / parent_wealth (same frame)")
    ax.set_title("Growth: wealth ratio distribution (same frame)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.0)

    fig.suptitle("Spawn statistics — same-frame vs prev-frame ratios", fontsize=14)
    plt.tight_layout()
    fig.savefig("spawn_stats3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved spawn_stats3.png")


if __name__ == "__main__":
    main()
