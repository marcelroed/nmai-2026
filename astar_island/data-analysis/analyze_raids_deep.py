"""
Deep raid analysis: quantify per-raid damage, pop survival, wealth stolen, takeover threshold.

Key insight from plots: defense delta shows diagonal structure, suggesting
proportional damage or multiple raids per step.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_replays import load_all_replays, build_transition_df
from sklearn.linear_model import LinearRegression

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

replays = load_all_replays()
df = build_transition_df(replays)

# ================================================================
# Step 1: Estimate natural defense recovery
# Use non-raided settlements (d_defense > 0 or very small drops)
# ================================================================
alive = df[(df["alive"]) & (df["alive_next"])].copy()
alive["owner_same"] = alive["owner_id"] == alive["owner_id_next"]

# Non-raided: d_defense >= -0.005 and same owner
non_raided = alive[(alive["d_defense"] >= -0.005) & (alive["owner_same"])].copy()
print(f"Non-raided transitions: {len(non_raided)}")

# Defense recovery model: d_defense = rate * (baseline - defense)
# This is linear in defense: d_defense = rate*baseline - rate*defense
reg_def = LinearRegression().fit(non_raided[["defense"]].values, non_raided["d_defense"].values)
recovery_rate = -reg_def.coef_[0]
baseline = reg_def.intercept_ / recovery_rate
print(f"\nDefense recovery:")
print(f"  Recovery rate: {recovery_rate:.4f}")
print(f"  Baseline: {baseline:.4f}")
print(f"  d_defense = {recovery_rate:.4f} * ({baseline:.4f} - defense)")
print(f"  R²: {reg_def.score(non_raided[['defense']].values, non_raided['d_defense'].values):.4f}")

# ================================================================
# Step 2: Identify raided settlements and compute raid damage
# ================================================================
# Raided: d_defense < -0.005 (excluding natural recovery)
# Compute expected recovery
alive["defense_recovery"] = recovery_rate * (baseline - alive["defense"])
alive["raw_damage"] = alive["d_defense"] - alive["defense_recovery"]

# Raided settlements: raw_damage < -0.005
raided = alive[alive["raw_damage"] < -0.005].copy()
print(f"\nRaided transitions: {len(raided)}")
print(f"  Ownership changed: {(~raided['owner_same']).sum()}")

# ================================================================
# Step 3: Analyze defense damage
# ================================================================
print(f"\n--- Defense damage analysis ---")
print(f"Raw damage stats:")
print(f"  mean={raided['raw_damage'].mean():.4f}, std={raided['raw_damage'].std():.4f}")
print(f"  median={raided['raw_damage'].median():.4f}")
print(f"  min={raided['raw_damage'].min():.4f}, max={raided['raw_damage'].max():.4f}")

# Is damage proportional to defense?
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].scatter(raided["defense"], raided["raw_damage"], alpha=0.15, s=3,
                   c=raided["owner_same"].map({True: "blue", False: "red"}))
axes[0, 0].set_xlabel("Defense (before raid)")
axes[0, 0].set_ylabel("Raw damage (d_defense - recovery)")
axes[0, 0].set_title("Raw damage vs defense")

# Check if damage = constant or proportional
# Try: damage = a * defense + b
reg_dmg = LinearRegression().fit(raided[["defense"]].values, raided["raw_damage"].values)
print(f"\nDamage ~ defense regression:")
print(f"  slope: {reg_dmg.coef_[0]:.4f}, intercept: {reg_dmg.intercept_:.4f}")
print(f"  R²: {reg_dmg.score(raided[['defense']].values, raided['raw_damage'].values):.4f}")
x_range = np.linspace(0, 1, 100)
axes[0, 0].plot(x_range, reg_dmg.predict(x_range.reshape(-1, 1)), "g-", linewidth=2)

# Distribution of raw damage
axes[0, 1].hist(raided["raw_damage"], bins=80, edgecolor="black", alpha=0.7)
axes[0, 1].set_xlabel("Raw damage")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Distribution of raw defense damage")

# Check for discrete damage levels (multiple raids)
# The damage histogram may show peaks at multiples of a base damage
# Let me look at the damage / defense ratio
raided["damage_ratio"] = raided["raw_damage"] / raided["defense"].clip(0.01)
axes[0, 2].hist(raided["damage_ratio"], bins=80, edgecolor="black", alpha=0.7)
axes[0, 2].set_xlabel("Damage / Defense")
axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title("Damage as fraction of defense")

# ================================================================
# Step 4: Pop survival analysis
# ================================================================
# Pop change = natural_growth + raid_effect
# Natural growth: pop_growth_rate * food * (1 - pop/pop_max)
# For the raided group, estimate natural growth from the model

# Fit pop growth on non-raided
non_raided["pop_growth_term"] = non_raided["food"] * (1 - non_raided["pop"] / 5.0)
reg_pop = LinearRegression().fit(
    non_raided[["pop_growth_term"]].values, non_raided["d_pop"].values
)
pop_growth_rate = reg_pop.coef_[0]
print(f"\nPop growth rate (from non-raided): {pop_growth_rate:.4f}")
print(f"  Intercept: {reg_pop.intercept_:.4f}")
print(f"  R²: {reg_pop.score(non_raided[['pop_growth_term']].values, non_raided['d_pop'].values):.4f}")

# Estimate natural pop growth for raided settlements
raided["expected_pop_growth"] = pop_growth_rate * raided["food"] * (1 - raided["pop"] / 5.0) + reg_pop.intercept_
raided["pop_after_growth"] = raided["pop"] + raided["expected_pop_growth"]
raided["pop_survival_ratio"] = raided["pop_next"] / raided["pop_after_growth"].clip(0.01)

print(f"\nPop survival ratio (pop_next / expected_pop_after_growth):")
print(f"  mean={raided['pop_survival_ratio'].mean():.4f}")
print(f"  std={raided['pop_survival_ratio'].std():.4f}")
print(f"  median={raided['pop_survival_ratio'].median():.4f}")

axes[1, 0].hist(raided["pop_survival_ratio"].clip(0.4, 1.2), bins=80, edgecolor="black", alpha=0.7)
axes[1, 0].set_xlabel("Pop survival ratio")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Pop survival ratio\n(accounting for natural growth)")

# Check if survival is multiplicative: pop_next = pop * (1-loss) + growth
# vs pop_next = (pop + growth) * (1-loss)
# The ordering matters! Let's check
raided["survival_v1"] = (raided["pop_next"] - raided["expected_pop_growth"]) / raided["pop"].clip(0.01)
raided["survival_v2"] = raided["pop_next"] / raided["pop_after_growth"].clip(0.01)

print(f"\nSurvival ratio v1 (pop*s + growth): mean={raided['survival_v1'].mean():.4f}, std={raided['survival_v1'].std():.4f}")
print(f"Survival ratio v2 ((pop+growth)*s): mean={raided['survival_v2'].mean():.4f}, std={raided['survival_v2'].std():.4f}")

# ================================================================
# Step 5: Wealth analysis
# ================================================================
# Wealth change for raided: d_wealth includes natural decay + raid theft
# Natural wealth change for non-raided
print(f"\nNon-raided wealth change: mean={non_raided['d_wealth'].mean():.4f}, std={non_raided['d_wealth'].std():.4f}")
print(f"Raided wealth change: mean={raided['d_wealth'].mean():.4f}, std={raided['d_wealth'].std():.4f}")

# Wealth stolen = raided_d_wealth - expected_natural_d_wealth
raided["wealth_stolen"] = raided["d_wealth"] - non_raided["d_wealth"].mean()
print(f"Estimated wealth stolen: mean={raided['wealth_stolen'].mean():.4f}")

# Is wealth stolen proportional to defender wealth?
reg_wealth = LinearRegression().fit(raided[["wealth"]].values, raided["wealth_stolen"].values)
print(f"Wealth stolen ~ wealth: slope={reg_wealth.coef_[0]:.4f}, intercept={reg_wealth.intercept_:.4f}")
print(f"  R²: {reg_wealth.score(raided[['wealth']].values, raided['wealth_stolen'].values):.4f}")

axes[1, 1].scatter(raided["wealth"], raided["wealth_stolen"], alpha=0.15, s=3)
x_r = np.linspace(0, raided["wealth"].max(), 100)
axes[1, 1].plot(x_r, reg_wealth.predict(x_r.reshape(-1, 1)), "r-", linewidth=2,
               label=f"slope={reg_wealth.coef_[0]:.3f}")
axes[1, 1].set_xlabel("Defender wealth (before)")
axes[1, 1].set_ylabel("Wealth stolen")
axes[1, 1].set_title("Wealth stolen vs defender wealth")
axes[1, 1].legend()

# ================================================================
# Step 6: Takeover threshold
# ================================================================
raided["defense_after"] = raided["defense_next"]
raided["success"] = ~raided["owner_same"]

# What defense level after raid leads to takeover?
def_bins = np.linspace(0, 0.4, 40)
raided["def_after_bin"] = pd.cut(raided["defense_after"], bins=def_bins)
takeover_by_def = raided.groupby("def_after_bin", observed=True).agg(
    p_takeover=("success", "mean"),
    n=("success", "count")
).reset_index()

axes[1, 2].bar(range(len(takeover_by_def)), takeover_by_def["p_takeover"], alpha=0.7)
axes[1, 2].set_xticks(range(0, len(takeover_by_def), 3))
tick_labels = [f"{x.mid:.2f}" for x in takeover_by_def["def_after_bin"]]
axes[1, 2].set_xticklabels(tick_labels[::3], rotation=45, fontsize=8)
axes[1, 2].set_xlabel("Defense after raid")
axes[1, 2].set_ylabel("P(takeover)")
axes[1, 2].set_title("Takeover probability vs post-raid defense")

# Find approximate threshold
successful = raided[raided["success"]]
print(f"\nTakeover analysis:")
print(f"  Defense after raid (successful): mean={successful['defense_after'].mean():.4f}, max={successful['defense_after'].max():.4f}")
print(f"  Defense after raid (unsuccessful): mean={raided[~raided['success']]['defense_after'].mean():.4f}")
threshold_candidates = raided[raided["success"]]["defense_after"]
print(f"  Max defense at takeover: {threshold_candidates.max():.4f}")
print(f"  95th percentile: {threshold_candidates.quantile(0.95):.4f}")
print(f"  99th percentile: {threshold_candidates.quantile(0.99):.4f}")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "raid_deep_analysis.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "raid_deep_analysis.png"), dpi=150)
print(f"\nSaved: {OUT}/raid_deep_analysis.pdf")

# ================================================================
# Step 7: Raid probability analysis
# ================================================================
print(f"\n{'='*60}")
print("Raid probability analysis")
print(f"{'='*60}")

# For each alive settlement, was it raided?
alive["was_raided"] = alive["raw_damage"] < -0.005

# Find nearest enemy for each settlement
# This requires going back to the replay data
# Let's use a simpler proxy: count of enemy neighbors
# Actually, let's just look at the probability by various factors

raid_rate = alive["was_raided"].mean()
print(f"Overall raid rate: {raid_rate:.4f}")

# By step
raid_by_step = alive.groupby("step")["was_raided"].mean()
print(f"\nRaid rate by step (first 5): {raid_by_step.head().to_dict()}")
print(f"Raid rate by step (last 5): {raid_by_step.tail().to_dict()}")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*60}")
print("RAID MECHANICS SUMMARY")
print(f"{'='*60}")
print(f"""
Defense:
  Recovery rate: {recovery_rate:.4f}
  Recovery baseline: {baseline:.4f}
  Raw damage mean: {raided['raw_damage'].mean():.4f}
  Damage appears to scale with defense (slope={reg_dmg.coef_[0]:.4f})

Population:
  Pop growth rate: {pop_growth_rate:.4f}
  Survival ratio (median): {raided['pop_survival_ratio'].median():.4f}

Wealth:
  Wealth stolen ∝ defender wealth (slope={reg_wealth.coef_[0]:.4f})
  Mean wealth stolen: {raided['wealth_stolen'].mean():.4f}

Takeover:
  Occurs when post-raid defense < ~{threshold_candidates.quantile(0.95):.3f}
  Rate: {raided['success'].mean():.4f} of raids result in takeover
""")
