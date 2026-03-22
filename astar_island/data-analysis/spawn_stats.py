"""
Investigate wealth and food values at settlement spawn.

A settlement "spawns" when it appears in frame i+1 but not in frame i.
We capture its stats at the first frame it appears.

Two spawn mechanisms:
1. Growth (phase 3): parent spawns child. child_food = parent.food * 0.1
2. Environment (phase 7): ruin reclaimed. food = nearest.food * 0.2

Questions:
1. What is the distribution of food at spawn?
2. What is the distribution of wealth at spawn?
3. Do these differ by spawn type (growth vs environment)?
4. Are they quantized / deterministic functions of parent state?

Reproduce:  python3 data-analysis/spawn_stats.py
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

    spawn_food = []
    spawn_wealth = []
    spawn_pop = []
    spawn_defense = []
    spawn_is_port = []
    spawn_step = []
    spawn_on_ruin = []  # True if spawned on a cell that was Ruin before

    # Also track potential parent stats
    parent_food = []  # food of nearest same-owner settlement in previous frame

    RUIN = 3
    SETTLEMENT = 1
    PORT = 2

    for replay in replays:
        frames = replay["frames"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            step = fa["step"]

            # Build set of alive settlement positions in frame before
            before_positions = set()
            before_map = {}
            for s in fb["settlements"]:
                if s["alive"]:
                    pos = (s["x"], s["y"])
                    before_positions.add(pos)
                    before_map[pos] = s

            # Find new settlements in frame after
            for s in fa["settlements"]:
                if not s["alive"]:
                    continue
                pos = (s["x"], s["y"])
                if pos in before_positions:
                    continue

                # This is a newly spawned settlement
                spawn_food.append(s["food"])
                spawn_wealth.append(s["wealth"])
                spawn_pop.append(s["population"])
                spawn_defense.append(s["defense"])
                spawn_is_port.append(s.get("has_port", False))
                spawn_step.append(step)

                # Check if the cell was a ruin in the previous frame
                x, y = pos
                grid_before = fb["grid"]
                was_ruin = grid_before[y][x] == RUIN
                spawn_on_ruin.append(was_ruin)

                # Find nearest same-owner settlement in before frame
                owner = s["owner_id"]
                min_dist = float("inf")
                nearest_food_val = None
                for bs_pos, bs in before_map.items():
                    if bs["owner_id"] != owner:
                        continue
                    dx = bs["x"] - x
                    dy = bs["y"] - y
                    dist = (dx*dx + dy*dy) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_food_val = bs["food"]
                parent_food.append(nearest_food_val)

    spawn_food = np.array(spawn_food)
    spawn_wealth = np.array(spawn_wealth)
    spawn_pop = np.array(spawn_pop)
    spawn_defense = np.array(spawn_defense)
    spawn_is_port = np.array(spawn_is_port)
    spawn_step = np.array(spawn_step)
    spawn_on_ruin = np.array(spawn_on_ruin)
    parent_food = np.array([f if f is not None else np.nan for f in parent_food])

    n = len(spawn_food)
    print(f"Total spawns: {n}")
    print(f"  On ruin: {spawn_on_ruin.sum()} ({100*spawn_on_ruin.mean():.1f}%)")
    print(f"  On non-ruin: {(~spawn_on_ruin).sum()} ({100*(~spawn_on_ruin).mean():.1f}%)")
    print(f"  Ports: {spawn_is_port.sum()} ({100*spawn_is_port.mean():.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # FOOD AT SPAWN
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FOOD AT SPAWN")
    print(f"{'='*70}")

    print(f"\n  All spawns:")
    print(f"    mean={spawn_food.mean():.6f}  std={spawn_food.std():.6f}")
    print(f"    min={spawn_food.min():.6f}  max={spawn_food.max():.6f}")
    print(f"    unique values: {len(np.unique(spawn_food))}")

    # Split by ruin vs non-ruin
    for label, mask in [("On ruin", spawn_on_ruin), ("On non-ruin", ~spawn_on_ruin)]:
        if mask.sum() == 0:
            continue
        f = spawn_food[mask]
        print(f"\n  {label} ({mask.sum()}):")
        print(f"    mean={f.mean():.6f}  std={f.std():.6f}")
        print(f"    min={f.min():.6f}  max={f.max():.6f}")
        print(f"    unique values: {len(np.unique(f))}")

        # Most common values
        vals, counts = np.unique(np.round(f, 6), return_counts=True)
        top = np.argsort(-counts)[:10]
        print(f"    top 10 values:")
        for j in top:
            print(f"      {vals[j]:.6f}  ({counts[j]} times, {100*counts[j]/mask.sum():.1f}%)")

    # Check if food = parent_food * factor
    has_parent = ~np.isnan(parent_food)
    if has_parent.sum() > 0:
        print(f"\n  Parent food available: {has_parent.sum()}")

        # For non-ruin spawns (growth): child_food = parent.food * 0.1
        mask_nr = (~spawn_on_ruin) & has_parent
        if mask_nr.sum() > 0:
            ratios = spawn_food[mask_nr] / parent_food[mask_nr]
            ratios = ratios[parent_food[mask_nr] > 0.01]
            print(f"\n  Growth spawns (non-ruin): food / parent_food")
            print(f"    mean={ratios.mean():.6f}  std={ratios.std():.6f}")
            print(f"    min={ratios.min():.6f}  max={ratios.max():.6f}")
            vals, counts = np.unique(np.round(ratios, 4), return_counts=True)
            top = np.argsort(-counts)[:10]
            print(f"    top 10 ratios:")
            for j in top:
                print(f"      {vals[j]:.4f}  ({counts[j]} times)")

        # For ruin spawns (environment): food = nearest.food * 0.2
        mask_r = spawn_on_ruin & has_parent
        if mask_r.sum() > 0:
            ratios = spawn_food[mask_r] / parent_food[mask_r]
            ratios = ratios[parent_food[mask_r] > 0.01]
            print(f"\n  Ruin spawns: food / nearest_food")
            print(f"    mean={ratios.mean():.6f}  std={ratios.std():.6f}")
            print(f"    min={ratios.min():.6f}  max={ratios.max():.6f}")
            vals, counts = np.unique(np.round(ratios, 4), return_counts=True)
            top = np.argsort(-counts)[:10]
            print(f"    top 10 ratios:")
            for j in top:
                print(f"      {vals[j]:.4f}  ({counts[j]} times)")

    # ═══════════════════════════════════════════════════════════════════
    # WEALTH AT SPAWN
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("WEALTH AT SPAWN")
    print(f"{'='*70}")

    print(f"\n  All spawns:")
    print(f"    mean={spawn_wealth.mean():.6f}  std={spawn_wealth.std():.6f}")
    print(f"    min={spawn_wealth.min():.6f}  max={spawn_wealth.max():.6f}")
    print(f"    unique values: {len(np.unique(spawn_wealth))}")

    vals, counts = np.unique(np.round(spawn_wealth, 6), return_counts=True)
    top = np.argsort(-counts)[:10]
    print(f"    top 10 values:")
    for j in top:
        print(f"      {vals[j]:.6f}  ({counts[j]} times, {100*counts[j]/n:.1f}%)")

    for label, mask in [("On ruin", spawn_on_ruin), ("On non-ruin", ~spawn_on_ruin)]:
        if mask.sum() == 0:
            continue
        w = spawn_wealth[mask]
        print(f"\n  {label} ({mask.sum()}):")
        print(f"    mean={w.mean():.6f}  std={w.std():.6f}")
        vals, counts = np.unique(np.round(w, 6), return_counts=True)
        top = np.argsort(-counts)[:5]
        print(f"    top values:")
        for j in top:
            print(f"      {vals[j]:.6f}  ({counts[j]} times, {100*counts[j]/mask.sum():.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # POPULATION AND DEFENSE AT SPAWN
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("POPULATION AT SPAWN")
    print(f"{'='*70}")

    for label, mask in [("On ruin", spawn_on_ruin), ("On non-ruin", ~spawn_on_ruin)]:
        if mask.sum() == 0:
            continue
        p = spawn_pop[mask]
        print(f"\n  {label} ({mask.sum()}):")
        vals, counts = np.unique(np.round(p, 4), return_counts=True)
        top = np.argsort(-counts)[:5]
        for j in top:
            print(f"    pop={vals[j]:.4f}  ({counts[j]} times, {100*counts[j]/mask.sum():.1f}%)")

    print(f"\n{'='*70}")
    print("DEFENSE AT SPAWN")
    print(f"{'='*70}")

    for label, mask in [("On ruin", spawn_on_ruin), ("On non-ruin", ~spawn_on_ruin)]:
        if mask.sum() == 0:
            continue
        d = spawn_defense[mask]
        print(f"\n  {label} ({mask.sum()}):")
        vals, counts = np.unique(np.round(d, 4), return_counts=True)
        top = np.argsort(-counts)[:5]
        for j in top:
            print(f"    def={vals[j]:.4f}  ({counts[j]} times, {100*counts[j]/mask.sum():.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Food histogram
    ax = axes[0, 0]
    ax.hist(spawn_food[~spawn_on_ruin], bins=80, alpha=0.6, color="steelblue",
            label=f"Growth ({(~spawn_on_ruin).sum()})", density=True)
    ax.hist(spawn_food[spawn_on_ruin], bins=80, alpha=0.6, color="orange",
            label=f"Ruin reclaim ({spawn_on_ruin.sum()})", density=True)
    ax.set_xlabel("food at spawn")
    ax.set_ylabel("density")
    ax.set_title("Food at spawn")
    ax.legend(fontsize=8)

    # Wealth histogram
    ax = axes[0, 1]
    ax.hist(spawn_wealth[~spawn_on_ruin], bins=80, alpha=0.6, color="steelblue",
            label="Growth", density=True)
    ax.hist(spawn_wealth[spawn_on_ruin], bins=80, alpha=0.6, color="orange",
            label="Ruin reclaim", density=True)
    ax.set_xlabel("wealth at spawn")
    ax.set_ylabel("density")
    ax.set_title("Wealth at spawn")
    ax.legend(fontsize=8)

    # Food vs parent food (growth)
    ax = axes[0, 2]
    mask_nr = (~spawn_on_ruin) & has_parent
    if mask_nr.sum() > 0:
        ax.scatter(parent_food[mask_nr], spawn_food[mask_nr], s=2, alpha=0.2,
                   color="steelblue", rasterized=True)
        lim = max(parent_food[mask_nr].max(), 1)
        ax.plot([0, lim], [0, lim*0.1], "r--", lw=1.5, label="y = 0.1x")
        ax.plot([0, lim], [0, lim*0.2], "g--", lw=1.5, label="y = 0.2x")
        ax.set_xlabel("parent food (nearest same-owner)")
        ax.set_ylabel("child food at spawn")
        ax.set_title("Growth: child food vs parent food")
        ax.legend(fontsize=8)

    # Food vs parent food (ruin reclaim)
    ax = axes[1, 0]
    mask_r = spawn_on_ruin & has_parent
    if mask_r.sum() > 0:
        ax.scatter(parent_food[mask_r], spawn_food[mask_r], s=5, alpha=0.3,
                   color="orange", rasterized=True)
        lim = max(parent_food[mask_r].max(), 1)
        ax.plot([0, lim], [0, lim*0.1], "r--", lw=1.5, label="y = 0.1x")
        ax.plot([0, lim], [0, lim*0.2], "g--", lw=1.5, label="y = 0.2x")
        ax.set_xlabel("nearest same-owner food")
        ax.set_ylabel("spawn food")
        ax.set_title("Ruin reclaim: spawn food vs nearest food")
        ax.legend(fontsize=8)

    # Pop histogram
    ax = axes[1, 1]
    ax.hist(spawn_pop[~spawn_on_ruin], bins=50, alpha=0.6, color="steelblue",
            label="Growth", density=True)
    ax.hist(spawn_pop[spawn_on_ruin], bins=50, alpha=0.6, color="orange",
            label="Ruin reclaim", density=True)
    ax.set_xlabel("population at spawn")
    ax.set_ylabel("density")
    ax.set_title("Population at spawn")
    ax.legend(fontsize=8)

    # Defense histogram
    ax = axes[1, 2]
    ax.hist(spawn_defense[~spawn_on_ruin], bins=50, alpha=0.6, color="steelblue",
            label="Growth", density=True)
    ax.hist(spawn_defense[spawn_on_ruin], bins=50, alpha=0.6, color="orange",
            label="Ruin reclaim", density=True)
    ax.set_xlabel("defense at spawn")
    ax.set_ylabel("density")
    ax.set_title("Defense at spawn")
    ax.legend(fontsize=8)

    fig.suptitle("Settlement spawn statistics", fontsize=14)
    plt.tight_layout()
    fig.savefig("spawn_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved spawn_stats.png")


if __name__ == "__main__":
    main()
