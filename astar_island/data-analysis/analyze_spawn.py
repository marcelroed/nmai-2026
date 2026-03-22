"""
Q2-Q4: Settlement spawning analysis
- What governs p_spawn?
- How is the parent influenced? What about the child?
- What is the terrain distribution of spawned settlements?
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_replays import load_all_replays, build_spawn_df, build_transition_df

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

replays = load_all_replays()
spawns = build_spawn_df(replays)
trans = build_transition_df(replays)

print(f"Total spawn events: {len(spawns)}")
print(f"Total transitions: {len(trans)}")

# ================================================================
# Q2: What governs p_spawn?
# ================================================================
# For each (round, seed, step), count how many settlements spawned
# and how many alive settlements existed (potential parents)
# p_spawn ≈ n_spawns / n_alive_settlements

alive_per_step = trans.groupby(["round_id", "seed", "step"]).size().reset_index(name="n_alive")
spawns_per_step = spawns.groupby(["round_id", "seed", "step"]).size().reset_index(name="n_spawns")

step_df = alive_per_step.merge(spawns_per_step, on=["round_id", "seed", "step"], how="left")
step_df["n_spawns"] = step_df["n_spawns"].fillna(0).astype(int)
step_df["p_spawn"] = step_df["n_spawns"] / step_df["n_alive"]

print(f"\n--- Spawn rate per step ---")
print(f"Mean p_spawn (per settlement): {step_df['p_spawn'].mean():.4f}")
print(f"Std p_spawn: {step_df['p_spawn'].std():.4f}")

# Now let's look at parent-level: which settlements spawn?
# For each alive settlement at step t, did it spawn a child at step t+1?
# We need to match parents to spawns
# The spawn df has parent info. Let's compute per-parent spawn probability
# as a function of parent pop and food.

# For each settlement-step, mark whether it spawned
parent_spawns = spawns[spawns["parent_dist"].notna()].copy()
# Count spawns per parent per step
parent_spawn_counts = parent_spawns.groupby(
    ["round_id", "seed", "step", "parent_x", "parent_y"]
).size().reset_index(name="n_children")

# Merge with transitions to get parent stats
trans_with_spawn = trans.merge(
    parent_spawn_counts,
    left_on=["round_id", "seed", "step", "x", "y"],
    right_on=["round_id", "seed", "step", "parent_x", "parent_y"],
    how="left"
)
trans_with_spawn["spawned"] = trans_with_spawn["n_children"].fillna(0).astype(int) > 0
trans_with_spawn["n_children"] = trans_with_spawn["n_children"].fillna(0).astype(int)

# Only alive settlements can spawn
alive = trans_with_spawn[trans_with_spawn["alive"]].copy()

print(f"\nTotal alive settlement-steps: {len(alive)}")
print(f"Spawned at least once: {alive['spawned'].sum()} ({alive['spawned'].mean():.4f})")

# --- Plot: spawn probability vs population and food ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Bin by population
alive["pop_bin"] = pd.cut(alive["pop"], bins=20)
spawn_by_pop = alive.groupby("pop_bin")["spawned"].agg(["mean", "count"]).reset_index()
axes[0, 0].bar(range(len(spawn_by_pop)), spawn_by_pop["mean"], alpha=0.7)
axes[0, 0].set_xticks(range(len(spawn_by_pop)))
axes[0, 0].set_xticklabels([f"{x.mid:.1f}" for x in spawn_by_pop["pop_bin"]], rotation=45, fontsize=7)
axes[0, 0].set_xlabel("Population")
axes[0, 0].set_ylabel("P(spawn)")
axes[0, 0].set_title("Spawn probability vs Population")

# Bin by food
alive["food_bin"] = pd.cut(alive["food"], bins=20)
spawn_by_food = alive.groupby("food_bin")["spawned"].agg(["mean", "count"]).reset_index()
axes[0, 1].bar(range(len(spawn_by_food)), spawn_by_food["mean"], alpha=0.7)
axes[0, 1].set_xticks(range(len(spawn_by_food)))
axes[0, 1].set_xticklabels([f"{x.mid:.2f}" for x in spawn_by_food["food_bin"]], rotation=45, fontsize=7)
axes[0, 1].set_xlabel("Food")
axes[0, 1].set_ylabel("P(spawn)")
axes[0, 1].set_title("Spawn probability vs Food")

# Bin by pop*food
alive["pop_x_food"] = alive["pop"] * alive["food"]
alive["pf_bin"] = pd.cut(alive["pop_x_food"], bins=20)
spawn_by_pf = alive.groupby("pf_bin")["spawned"].agg(["mean", "count"]).reset_index()
axes[0, 2].bar(range(len(spawn_by_pf)), spawn_by_pf["mean"], alpha=0.7)
axes[0, 2].set_xticks(range(len(spawn_by_pf)))
axes[0, 2].set_xticklabels([f"{x.mid:.1f}" for x in spawn_by_pf["pf_bin"]], rotation=45, fontsize=7)
axes[0, 2].set_xlabel("Population × Food")
axes[0, 2].set_ylabel("P(spawn)")
axes[0, 2].set_title("Spawn probability vs Pop × Food")

# 2D heatmap: pop vs food → p_spawn
alive["pop_bin2"] = pd.cut(alive["pop"], bins=10)
alive["food_bin2"] = pd.cut(alive["food"], bins=10)
heatmap = alive.groupby(["pop_bin2", "food_bin2"])["spawned"].mean().unstack()
im = axes[1, 0].imshow(heatmap.values, aspect="auto", origin="lower", cmap="YlOrRd")
axes[1, 0].set_xlabel("Food bin")
axes[1, 0].set_ylabel("Pop bin")
axes[1, 0].set_title("P(spawn) heatmap: Pop × Food")
plt.colorbar(im, ax=axes[1, 0])

# Spawn probability vs step
spawn_by_step = alive.groupby("step")["spawned"].mean()
axes[1, 1].plot(spawn_by_step.index, spawn_by_step.values, ".-")
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("P(spawn)")
axes[1, 1].set_title("Spawn probability vs Step")

# Spawn probability vs has_port
spawn_by_port = alive.groupby("has_port")["spawned"].mean()
axes[1, 2].bar(["No Port", "Port"], spawn_by_port.values, alpha=0.7)
axes[1, 2].set_ylabel("P(spawn)")
axes[1, 2].set_title("Spawn probability vs Port status")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "spawn_probability.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "spawn_probability.png"), dpi=150)
print(f"Saved: {OUT}/spawn_probability.pdf")

# ================================================================
# Q3: Parent/child effects on spawn
# ================================================================
print("\n" + "="*60)
print("Q3: Parent/Child effects")
print("="*60)

# Child initial stats
print(f"\nChild stats at birth:")
for col in ["child_pop", "child_food", "child_wealth", "child_defense"]:
    print(f"  {col}: mean={spawns[col].mean():.4f}, std={spawns[col].std():.4f}, "
          f"min={spawns[col].min():.4f}, max={spawns[col].max():.4f}")
    # Unique values
    uniq = sorted(spawns[col].unique())
    if len(uniq) <= 20:
        print(f"    Unique values: {uniq}")

# Child stats by terrain before
print(f"\nChild stats by terrain before spawn:")
for terrain, g in spawns.groupby("terrain_before_name"):
    if len(g) < 5:
        continue
    print(f"  Terrain: {terrain} (n={len(g)})")
    for col in ["child_pop", "child_food", "child_wealth", "child_defense"]:
        print(f"    {col}: mean={g[col].mean():.4f}, std={g[col].std():.4f}")

# Parent delta (compare parent stats before and after spawn)
has_parent = spawns[spawns["parent_pop"].notna()].copy()
has_parent["parent_d_pop"] = has_parent["parent_pop_next"] - has_parent["parent_pop"]
has_parent["parent_d_food"] = has_parent["parent_food_next"] - has_parent["parent_food"]
has_parent["parent_d_wealth"] = has_parent["parent_wealth_next"] - has_parent["parent_wealth"]
has_parent["parent_d_defense"] = has_parent["parent_defense_next"] - has_parent["parent_defense"]

print(f"\nParent delta when spawning (raw, includes food/pop growth effects):")
for col in ["parent_d_pop", "parent_d_food", "parent_d_wealth", "parent_d_defense"]:
    print(f"  {col}: mean={has_parent[col].mean():.4f}, std={has_parent[col].std():.4f}")

# Compare to non-spawning settlements
non_spawning = alive[~alive["spawned"]].copy()
print(f"\nNon-spawning settlements delta (baseline):")
for col in ["d_pop", "d_food", "d_wealth", "d_defense"]:
    print(f"  {col}: mean={non_spawning[col].mean():.4f}, std={non_spawning[col].std():.4f}")

spawning = alive[alive["spawned"]].copy()
print(f"\nSpawning settlements delta:")
for col in ["d_pop", "d_food", "d_wealth", "d_defense"]:
    print(f"  {col}: mean={spawning[col].mean():.4f}, std={spawning[col].std():.4f}")

# --- Plot: parent and child effects ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Child pop distribution
axes[0, 0].hist(spawns["child_pop"], bins=50, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Child population at birth")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Child population distribution")

# Child defense distribution
axes[0, 1].hist(spawns["child_defense"], bins=50, edgecolor="black", alpha=0.7)
axes[0, 1].set_xlabel("Child defense at birth")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Child defense distribution")

# Child food distribution
axes[0, 2].hist(spawns["child_food"], bins=50, edgecolor="black", alpha=0.7)
axes[0, 2].set_xlabel("Child food at birth")
axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title("Child food distribution")

# Parent pop delta: spawning vs non-spawning
axes[1, 0].hist(non_spawning["d_pop"].clip(-1, 1), bins=80, alpha=0.5, label="Non-spawning", density=True)
axes[1, 0].hist(spawning["d_pop"].clip(-1, 1), bins=80, alpha=0.5, label="Spawning", density=True)
axes[1, 0].set_xlabel("Population delta")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title("Pop delta: spawning vs non-spawning parents")
axes[1, 0].legend()

# Parent food delta
axes[1, 1].hist(non_spawning["d_food"].clip(-0.5, 0.5), bins=80, alpha=0.5, label="Non-spawning", density=True)
axes[1, 1].hist(spawning["d_food"].clip(-0.5, 0.5), bins=80, alpha=0.5, label="Spawning", density=True)
axes[1, 1].set_xlabel("Food delta")
axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("Food delta: spawning vs non-spawning parents")
axes[1, 1].legend()

# Parent distance to child
axes[1, 2].hist(has_parent["parent_dist"], bins=range(0, 10), edgecolor="black", alpha=0.7)
axes[1, 2].set_xlabel("Chebyshev distance (parent → child)")
axes[1, 2].set_ylabel("Count")
axes[1, 2].set_title("Parent-child distance")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "spawn_effects.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "spawn_effects.png"), dpi=150)
print(f"Saved: {OUT}/spawn_effects.pdf")

# ================================================================
# Q4: Terrain distribution of spawned settlements
# ================================================================
print("\n" + "="*60)
print("Q4: Terrain distribution of spawned settlements")
print("="*60)

terrain_counts = spawns["terrain_before_name"].value_counts()
print(f"\nTerrain before spawn:")
for terrain, count in terrain_counts.items():
    print(f"  {terrain}: {count} ({count/len(spawns)*100:.1f}%)")

# Port status of children
port_rate = spawns["child_has_port"].mean()
print(f"\nChild is port: {port_rate:.4f} ({spawns['child_has_port'].sum()} / {len(spawns)})")
print(f"Port children adjacent to ocean: {spawns[spawns['child_has_port']]['n_adj_ocean'].mean():.2f} (should be > 0)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Terrain pie chart
axes[0].pie(terrain_counts.values, labels=terrain_counts.index, autopct="%1.1f%%")
axes[0].set_title("Terrain before settlement spawn")

# Child port status by ocean adjacency
ocean_bins = spawns.groupby("n_adj_ocean")["child_has_port"].agg(["mean", "count"]).reset_index()
axes[1].bar(ocean_bins["n_adj_ocean"], ocean_bins["mean"], alpha=0.7)
for i, row in ocean_bins.iterrows():
    axes[1].text(row["n_adj_ocean"], row["mean"] + 0.01, f"n={int(row['count'])}", ha="center", fontsize=8)
axes[1].set_xlabel("# Adjacent ocean tiles")
axes[1].set_ylabel("P(child is port)")
axes[1].set_title("Port probability by ocean adjacency")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "spawn_terrain.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "spawn_terrain.png"), dpi=150)
print(f"Saved: {OUT}/spawn_terrain.pdf")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nSpawn probability (per settlement per step): ~{alive['spawned'].mean():.4f}")
print(f"Spawn probability appears to scale with population × food")
print(f"\nChild stats (by terrain):")
for terrain, g in spawns.groupby("terrain_before_name"):
    if len(g) >= 5:
        print(f"  {terrain} (n={len(g)}): pop={g['child_pop'].mean():.3f}, def={g['child_defense'].mean():.3f}, "
              f"food={g['child_food'].mean():.3f}, wealth={g['child_wealth'].mean():.3f}")
print(f"\nTerrain distribution: {dict(terrain_counts)}")
print(f"Port rate: {port_rate:.4f}")
