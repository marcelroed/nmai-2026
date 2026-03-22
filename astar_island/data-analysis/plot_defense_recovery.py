"""
Scatter plot: population (before step) vs defense_delta, one subplot per round.

Reproduce:  python3 data-analysis/plot_defense_recovery.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

POP_RATIO_MAX = 1.2
POP_RATIO_MIN = 0.85
DEF_CAP = 0.999


def load_replays_by_round():
    by_round: dict[str, list] = defaultdict(list)
    for rd in sorted(DATA_DIR.iterdir()):
        if not rd.is_dir():
            continue
        ad = rd / "analysis"
        if not ad.exists():
            continue
        for f in sorted(ad.glob("replay_seed_index=*.json")):
            with open(f) as fh:
                data = json.load(fh)
            by_round[data["round_id"]].append(data)
    return dict(by_round)


def collect_rows(replays):
    """Return list of (pop_before, def_delta).

    Filters to defense-recovery events only:
    - Settlement alive in both frames, same owner
    - def_delta > 0 (defense increased)
    - Population ratio (after/before) in [0.85, 1.2] (roughly stable pop)
    - Post-step defense < 0.999 (not at cap)
    """
    rows = []
    for replay in replays:
        frames = replay["frames"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            before_map = {
                (s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]
            }
            after_map = {}
            for s in fa["settlements"]:
                k = (s["x"], s["y"])
                if k not in after_map or s["alive"]:
                    after_map[k] = s

            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or not sa["alive"]:
                    continue
                if sa["owner_id"] != sb["owner_id"]:
                    continue
                def_delta = sa["defense"] - sb["defense"]
                if def_delta <= 0:
                    continue
                pop_ratio = sa["population"] / sb["population"] if sb["population"] > 0.01 else 99
                if pop_ratio > POP_RATIO_MAX or pop_ratio < POP_RATIO_MIN:
                    continue
                if sa["defense"] >= DEF_CAP:
                    continue
                rows.append((sb["population"], def_delta))
    return rows


def main():
    rounds = load_replays_by_round()
    round_ids = sorted(rounds.keys())
    n = len(round_ids)
    cols = 3
    rows_grid = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_grid, cols, figsize=(5 * cols, 4 * rows_grid),
                             squeeze=False)

    for idx, rid in enumerate(round_ids):
        ax = axes[idx // cols][idx % cols]
        data = collect_rows(rounds[rid])
        if not data:
            continue
        arr = np.array(data)
        pop_before = arr[:, 0]
        delta = arr[:, 1]

        # fit line
        c = np.dot(pop_before, delta) / np.dot(pop_before, pop_before)

        ax.scatter(pop_before, delta, s=1, alpha=0.15, color="steelblue",
                   rasterized=True)

        x_line = np.array([0, pop_before.max()])
        ax.plot(x_line, c * x_line, color="red", linewidth=1.5,
                label=f"c={c:.5f}")

        ax.set_title(f"{rid[:12]}  (n={len(data)})", fontsize=10)
        ax.set_xlabel("pop_before", fontsize=8)
        ax.set_ylabel("def_delta", fontsize=8)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for idx in range(n, rows_grid * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Defense recovery: pop_before vs def_delta (per round)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(__file__).resolve().parent / "defense_recovery_scatter.png"
    fig.savefig(out, dpi=180)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
