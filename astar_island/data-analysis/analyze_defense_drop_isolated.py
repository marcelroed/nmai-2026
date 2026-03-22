"""Investigate settlements that experience a defense drop (defense_{t+1} < defense_t)
and have exactly one other settlement within taxicab distance 6 at time t.

Plots the distribution of delta_wealth for those events.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_all_replays():
    replays = []
    for path in sorted(DATA_DIR.glob("*/analysis/replay_seed_index=*.json")):
        with open(path) as f:
            d = json.load(f)
        replays.append((d["round_id"], d["seed_index"], d["frames"]))
    print(f"Loaded {len(replays)} replays")
    return replays


def find_defense_drop_isolated(replays):
    """Find transitions where defense drops and settlement has exactly 1 neighbor within taxicab dist 6."""
    events = []

    for round_id, seed, frames in replays:
        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            step = f0["step"]

            s0_map = {(s["x"], s["y"]): s for s in f0["settlements"]}
            s1_map = {(s["x"], s["y"]): s for s in f1["settlements"]}

            alive_positions = [(s["x"], s["y"]) for s in f0["settlements"] if s["alive"]]

            for (x, y), s in s0_map.items():
                if not s["alive"]:
                    continue
                if (x, y) not in s1_map:
                    continue
                s1 = s1_map[(x, y)]

                # Defense must drop
                if s1["defense"] >= s["defense"]:
                    continue

                # Find other alive settlements within taxicab distance 6
                neighbors = []
                for ax, ay in alive_positions:
                    if ax == x and ay == y:
                        continue
                    if abs(ax - x) + abs(ay - y) <= 6:
                        neighbors.append((ax, ay))

                if len(neighbors) != 1:
                    continue

                # Get neighbor's delta_wealth
                nx, ny = neighbors[0]
                nb0 = s0_map[(nx, ny)]
                nb1 = s1_map.get((nx, ny))
                nb_d_wealth = (nb1["wealth"] - nb0["wealth"]) if nb1 else -nb0["wealth"]
                nb_d_defense = (nb1["defense"] - nb0["defense"]) if nb1 else -nb0["defense"]

                events.append({
                    "round_id": round_id[:8],
                    "seed": seed,
                    "step": step,
                    "x": x, "y": y,
                    "defense_t": s["defense"],
                    "defense_t1": s1["defense"],
                    "d_defense": s1["defense"] - s["defense"],
                    "wealth_t": s["wealth"],
                    "wealth_t1": s1["wealth"],
                    "d_wealth": s1["wealth"] - s["wealth"],
                    "pop_t": s["population"],
                    "pop_t1": s1["population"],
                    "d_pop": s1["population"] - s["population"],
                    "owner_id": s["owner_id"],
                    "owner_changed": s1["owner_id"] != s["owner_id"],
                    "nb_x": nx, "nb_y": ny,
                    "nb_owner_id": nb0["owner_id"],
                    "nb_same_owner": nb0["owner_id"] == s["owner_id"],
                    "nb_wealth_t": nb0["wealth"],
                    "nb_wealth_t1": nb1["wealth"] if nb1 else 0,
                    "nb_d_wealth": nb_d_wealth,
                    "nb_pop_t": nb0["population"],
                    "nb_defense_t": nb0["defense"],
                    "nb_d_defense": nb_d_defense,
                })

    print(f"Found {len(events)} defense-drop events with exactly 1 neighbor within taxicab dist 6")
    return events


def fit_defense_growth(replays):
    """Fit c per round such that def_growth = c * pop, using transitions where ddef > 0."""
    # Collect (round_id[:8], ddef, pop) for all positive-ddef transitions
    per_round = {}
    for round_id, seed, frames in replays:
        rid = round_id[:8]
        if rid not in per_round:
            per_round[rid] = {"ddef": [], "pop": []}
        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i + 1]
            s0_map = {(s["x"], s["y"]): s for s in f0["settlements"]}
            s1_map = {(s["x"], s["y"]): s for s in f1["settlements"]}
            for (x, y), s in s0_map.items():
                if not s["alive"] or (x, y) not in s1_map:
                    continue
                s1 = s1_map[(x, y)]
                dd = s1["defense"] - s["defense"]
                if dd > 0:
                    per_round[rid]["ddef"].append(dd)
                    per_round[rid]["pop"].append(s["population"])

    c_per_round = {}
    for rid, data in per_round.items():
        ddef = np.array(data["ddef"])
        pop = np.array(data["pop"])
        # Fit c via least squares: ddef = c * pop (no intercept)
        c = np.sum(ddef * pop) / np.sum(pop * pop)
        c_per_round[rid] = c
        print(f"  Round {rid}: c={c:.5f} (n={len(ddef)})")
    return c_per_round


def main():
    replays = load_all_replays()
    c_per_round = fit_defense_growth(replays)
    events = find_defense_drop_isolated(replays)

    if not events:
        print("No matching events found.")
        return

    d_wealth = np.array([e["d_wealth"] for e in events])
    d_defense = np.array([e["d_defense"] for e in events])
    owner_changed = np.array([e["owner_changed"] for e in events])

    print(f"\n--- Summary ---")
    print(f"Total events: {len(events)}")
    print(f"  Owner changed: {owner_changed.sum()}")
    print(f"  Owner kept:    {(~owner_changed).sum()}")
    print(f"\ndelta_wealth stats:")
    print(f"  mean:   {d_wealth.mean():.4f}")
    print(f"  median: {np.median(d_wealth):.4f}")
    print(f"  std:    {d_wealth.std():.4f}")
    print(f"  min:    {d_wealth.min():.4f}")
    print(f"  max:    {d_wealth.max():.4f}")
    print(f"\ndelta_defense stats:")
    print(f"  mean:   {d_defense.mean():.4f}")
    print(f"  median: {np.median(d_defense):.4f}")

    nb_d_wealth = np.array([e["nb_d_wealth"] for e in events])
    nb_same_owner = np.array([e["nb_same_owner"] for e in events])

    print(f"\nneighbor delta_wealth stats:")
    print(f"  mean:   {nb_d_wealth.mean():.4f}")
    print(f"  median: {np.median(nb_d_wealth):.4f}")
    print(f"  Same owner neighbors: {nb_same_owner.sum()}")
    print(f"  Diff owner neighbors: {(~nb_same_owner).sum()}")

    # --- Defense drop distribution plot ---
    fig_def, ax_def = plt.subplots(figsize=(10, 6))
    ax_def.hist(d_defense, bins=80, edgecolor="black", alpha=0.7)
    ax_def.set_xlabel("delta_defense")
    ax_def.set_ylabel("Count")
    ax_def.set_title(f"Defense drop distribution (N={len(events)})\n"
                     f"Settlements with exactly 1 neighbor within taxicab dist 6")
    # Mark quantiles
    for q, ls in [(0.25, ":"), (0.5, "--"), (0.75, ":")]:
        val = np.quantile(d_defense, q)
        ax_def.axvline(val, color="red", linestyle=ls, linewidth=1,
                       label=f"Q{int(q*100)}={val:.3f}")
    ax_def.legend()
    plt.tight_layout()
    out_def = Path(__file__).parent.parent / "data-analysis" / "defense_drop_distribution.png"
    fig_def.savefig(out_def, dpi=150)
    print(f"\nSaved defense drop plot to {out_def}")

    # --- Shared masks ---
    same = nb_same_owner
    diff = ~nb_same_owner

    # --- Defense drop correlates (subtract growth) ---
    own_def = np.array([e["defense_t"] for e in events])
    own_pop = np.array([e["pop_t"] for e in events])
    nb_pop = np.array([e["nb_pop_t"] for e in events])
    nb_def = np.array([e["nb_defense_t"] for e in events])

    # Compute dgrowth = c * pop per event, then raid_damage = ddef - dgrowth
    dgrowth = np.array([c_per_round[e["round_id"]] * e["pop_t"] for e in events])
    d_raid = d_defense - dgrowth

    print(f"\nRaid damage (ddef - dgrowth) stats:")
    print(f"  mean:   {d_raid.mean():.4f}")
    print(f"  median: {np.median(d_raid):.4f}")
    print(f"  std:    {d_raid.std():.4f}")

    fig_corr, axes_corr = plt.subplots(2, 3, figsize=(18, 10))
    fig_corr.suptitle(f"What drives the defense drop? (N={diff.sum()} diff-owner raids)\n"
                      f"y-axis = ddef - c*pop (growth subtracted)", fontsize=14)

    # 1) raid damage vs own defense_t
    ax = axes_corr[0, 0]
    ax.scatter(own_def[diff], d_raid[diff], alpha=0.4, s=15)
    # Reference lines: raid_damage = -k * def
    x_line = np.linspace(0, own_def[diff].max(), 100)
    for k in [0.20, 0.30, 0.44]:
        ax.plot(x_line, -k * x_line, linestyle="--", linewidth=1.5, alpha=0.7, label=f"-{k} * def")
    ax.set_xlabel("self.defense_t")
    ax.set_ylabel("ddef - dgrowth")
    ax.set_title("vs own defense")
    r = np.corrcoef(own_def[diff], d_raid[diff])[0, 1]
    ax.annotate(f"r={r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")
    ax.legend(fontsize=8)

    # 2) raid damage vs own population_t
    ax = axes_corr[0, 1]
    ax.scatter(own_pop[diff], d_raid[diff], alpha=0.4, s=15)
    ax.set_xlabel("self.population_t")
    ax.set_ylabel("ddef - dgrowth")
    ax.set_title("vs own population")
    r = np.corrcoef(own_pop[diff], d_raid[diff])[0, 1]
    ax.annotate(f"r={r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")

    # 3) raid damage vs neighbor defense_t
    ax = axes_corr[0, 2]
    ax.scatter(nb_def[diff], d_raid[diff], alpha=0.4, s=15)
    ax.set_xlabel("attacker.defense_t")
    ax.set_ylabel("ddef - dgrowth")
    ax.set_title("vs attacker defense")
    r = np.corrcoef(nb_def[diff], d_raid[diff])[0, 1]
    ax.annotate(f"r={r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")

    # 4) raid damage vs d_wealth
    ax = axes_corr[1, 0]
    ax.scatter(d_wealth[diff], d_raid[diff], alpha=0.4, s=15)
    ax.set_xlabel("self.delta_wealth")
    ax.set_ylabel("ddef - dgrowth")
    ax.set_title("vs own delta_wealth")
    r = np.corrcoef(d_wealth[diff], d_raid[diff])[0, 1]
    ax.annotate(f"r={r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")

    # 5) raid damage vs neighbor d_wealth (attacker wealth gain)
    ax = axes_corr[1, 1]
    ax.scatter(nb_d_wealth[diff], d_raid[diff], alpha=0.4, s=15)
    ax.set_xlabel("attacker.delta_wealth")
    ax.set_ylabel("ddef - dgrowth")
    ax.set_title("vs attacker delta_wealth")
    r = np.corrcoef(nb_d_wealth[diff], d_raid[diff])[0, 1]
    ax.annotate(f"r={r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")

    # 6) raid damage vs attacker population
    ax = axes_corr[1, 2]
    ax.scatter(nb_pop[diff], d_raid[diff], alpha=0.4, s=15)
    ax.set_xlabel("attacker.population_t")
    ax.set_ylabel("ddef - dgrowth")
    ax.set_title("vs attacker population")
    r = np.corrcoef(nb_pop[diff], d_raid[diff])[0, 1]
    ax.annotate(f"r={r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")

    plt.tight_layout()
    out_corr = Path(__file__).parent.parent / "data-analysis" / "defense_drop_correlates.png"
    fig_corr.savefig(out_corr, dpi=150)
    print(f"Saved correlates plot to {out_corr}")

    # --- delta_wealth vs own defense scatter ---
    fig_wd, ax_wd = plt.subplots(figsize=(10, 7))
    ax_wd.scatter(own_def[diff], d_wealth[diff], alpha=0.4, s=15, label=f"Diff owner (n={diff.sum()})")
    ax_wd.scatter(own_def[same], d_wealth[same], alpha=0.4, s=15, label=f"Same owner (n={same.sum()})")
    ax_wd.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    r = np.corrcoef(own_def[diff], d_wealth[diff])[0, 1]
    ax_wd.annotate(f"r={r:.3f} (diff-owner)", xy=(0.05, 0.95), xycoords="axes fraction", va="top", fontsize=12)
    ax_wd.set_xlabel("self.defense_t", fontsize=12)
    ax_wd.set_ylabel("self.delta_wealth", fontsize=12)
    ax_wd.set_title("delta_wealth vs own defense at time of raid", fontsize=14)
    ax_wd.legend()
    plt.tight_layout()
    out_wd = Path(__file__).parent.parent / "data-analysis" / "dwealth_vs_own_defense.png"
    fig_wd.savefig(out_wd, dpi=150)
    print(f"Saved delta_wealth vs defense plot to {out_wd}")

    # --- Wealth plots (existing) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Histogram of delta_wealth
    ax = axes[0, 0]
    ax.hist(d_wealth, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("delta_wealth")
    ax.set_ylabel("Count")
    ax.set_title(f"delta_wealth distribution (N={len(events)})\n"
                 f"defense drop + exactly 1 neighbor within taxicab dist 6")

    # 2) Split by owner change
    ax = axes[0, 1]
    d_w_changed = d_wealth[owner_changed]
    d_w_kept = d_wealth[~owner_changed]
    bins = np.linspace(d_wealth.min(), d_wealth.max(), 50)
    ax.hist(d_w_kept, bins=bins, alpha=0.6, label=f"Owner kept (n={len(d_w_kept)})", edgecolor="black")
    ax.hist(d_w_changed, bins=bins, alpha=0.6, label=f"Owner changed (n={len(d_w_changed)})", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("delta_wealth")
    ax.set_ylabel("Count")
    ax.set_title("delta_wealth by ownership change")
    ax.legend()

    # 3) self.delta_wealth vs neighbor.delta_wealth scatter
    ax = axes[1, 0]
    ax.scatter(d_wealth[diff], nb_d_wealth[diff],
               alpha=0.4, s=15, label=f"Diff owner (n={diff.sum()})", zorder=2)
    ax.scatter(d_wealth[same], nb_d_wealth[same],
               alpha=0.4, s=15, label=f"Same owner (n={same.sum()})", zorder=2)
    # Reference lines
    lim_min = min(d_wealth.min(), nb_d_wealth.min()) - 0.02
    lim_max = max(d_wealth.max(), nb_d_wealth.max()) + 0.02
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="red", linestyle=":", linewidth=1, label="y=x")
    ax.set_xlabel("self.delta_wealth")
    ax.set_ylabel("neighbor.delta_wealth")
    ax.set_title("self vs neighbor delta_wealth")
    ax.legend()

    # 4) delta_wealth by step
    steps = np.array([e["step"] for e in events])
    ax = axes[1, 1]
    ax.scatter(steps, d_wealth, alpha=0.3, s=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("delta_wealth")
    ax.set_title("delta_wealth over simulation time")

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "data-analysis" / "defense_drop_isolated_wealth.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved wealth plot to {out_path}")

    # --- Newly spawned settlement defense distribution ---
    plot_spawn_defense(replays)


def find_spawns(replays):
    """Find settlements that just appeared: present at t+1 but not at t."""
    spawns = []
    for round_id, seed, frames in replays:
        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            s0_pos = {(s["x"], s["y"]) for s in f0["settlements"]}
            for s1 in f1["settlements"]:
                if not s1["alive"]:
                    continue
                pos = (s1["x"], s1["y"])
                if pos not in s0_pos:
                    spawns.append({
                        "round_id": round_id[:8],
                        "seed": seed,
                        "step": f0["step"],
                        "x": s1["x"], "y": s1["y"],
                        "defense": s1["defense"],
                        "population": s1["population"],
                        "wealth": s1["wealth"],
                        "food": s1["food"],
                        "has_port": s1["has_port"],
                    })
    print(f"Found {len(spawns)} newly spawned settlements")
    return spawns


def plot_spawn_defense(replays):
    spawns = find_spawns(replays)
    if not spawns:
        print("No spawns found.")
        return

    defense = np.array([s["defense"] for s in spawns])

    print(f"\nSpawn defense stats:")
    print(f"  mean:   {defense.mean():.4f}")
    print(f"  median: {np.median(defense):.4f}")
    print(f"  std:    {defense.std():.4f}")
    print(f"  min:    {defense.min():.4f}")
    print(f"  max:    {defense.max():.4f}")

    # Check unique values
    unique, counts = np.unique(np.round(defense, 4), return_counts=True)
    if len(unique) <= 30:
        print(f"\nUnique defense values:")
        for v, c in sorted(zip(unique, counts), key=lambda t: -t[1]):
            print(f"  {v:.4f}: {c} ({100*c/len(defense):.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(defense, bins=80, edgecolor="black", alpha=0.7)
    ax.set_xlabel("defense")
    ax.set_ylabel("Count")
    ax.set_title(f"Defense of newly spawned settlements (N={len(spawns)})")
    for q, ls in [(0.25, ":"), (0.5, "--"), (0.75, ":")]:
        val = np.quantile(defense, q)
        ax.axvline(val, color="red", linestyle=ls, linewidth=1,
                   label=f"Q{int(q*100)}={val:.3f}")
    ax.legend()
    plt.tight_layout()
    out = Path(__file__).parent.parent / "data-analysis" / "spawn_defense_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"Saved spawn defense plot to {out}")


if __name__ == "__main__":
    main()
