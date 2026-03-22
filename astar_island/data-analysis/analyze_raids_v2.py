"""
Raid analysis v2: Clean approach using raw defense drops.

Key change: use d_defense < -0.01 as raid criterion, don't try to model recovery.
Focus on what the data clearly shows.
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

alive = df[(df["alive"]) & (df["alive_next"])].copy()
alive["owner_same"] = alive["owner_id"] == alive["owner_id_next"]

# ================================================================
# 1. Defense change distribution for all settlements
# ================================================================
print("=== Defense change distribution ===")
for thresh in [0, -0.005, -0.01, -0.02, -0.05]:
    n = (alive["d_defense"] < thresh).sum()
    print(f"  d_defense < {thresh:.3f}: {n} ({n/len(alive)*100:.1f}%)")

# Non-raided: d_defense >= 0
non_raided = alive[(alive["d_defense"] >= -0.001) & alive["owner_same"]].copy()
print(f"\nNon-raided (d_defense >= -0.001): {len(non_raided)}")
print(f"  Mean d_defense: {non_raided['d_defense'].mean():.6f}")
print(f"  Std d_defense: {non_raided['d_defense'].std():.6f}")
print(f"  Median d_defense: {non_raided['d_defense'].median():.6f}")

# Defense recovery: look at d_defense vs defense for non-raided
# Bin by defense
non_raided["def_bin"] = pd.cut(non_raided["defense"], bins=20)
recovery_by_def = non_raided.groupby("def_bin", observed=True)["d_defense"].agg(["mean", "std", "count"])
print(f"\nDefense recovery by defense level:")
print(recovery_by_def.to_string())

# ================================================================
# 2. Clearly raided: d_defense < -0.01
# ================================================================
RAID_THRESH = -0.01
raided = alive[alive["d_defense"] < RAID_THRESH].copy()
print(f"\n=== Raided (d_defense < {RAID_THRESH}) ===")
print(f"Count: {len(raided)} ({len(raided)/len(alive)*100:.1f}%)")
print(f"Ownership changed: {(~raided['owner_same']).sum()}")

# ================================================================
# 3. Defense damage quantification
# ================================================================
print(f"\n--- Defense damage ---")
print(f"d_defense: mean={raided['d_defense'].mean():.4f}, median={raided['d_defense'].median():.4f}")
print(f"  std={raided['d_defense'].std():.4f}")

# Look for structure: is damage clustered?
# Round to 3 decimal places and look at most common values
rounded = raided["d_defense"].round(3)
print(f"\nMost common defense deltas:")
print(rounded.value_counts().head(15))

# Damage distribution plot
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].hist(raided["d_defense"], bins=100, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Defense delta")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Raided: defense delta distribution")

# Scatter: d_defense vs defense (the key plot)
axes[0, 1].scatter(raided["defense"], raided["d_defense"], alpha=0.05, s=2,
                   c=raided["owner_same"].map({True: "blue", False: "red"}))
axes[0, 1].set_xlabel("Defense (before)")
axes[0, 1].set_ylabel("Defense delta")
axes[0, 1].set_title("Defense delta vs defense\n(blue=survived, red=taken)")

# ================================================================
# 4. Pop analysis for raided settlements
# ================================================================
# To isolate raid pop effect, compare to "expected" pop from non-raided
# Non-raided pop model
non_raided["logistic"] = non_raided["food"] * (1 - non_raided["pop"] / 5.0)
features_pop = ["pop", "logistic"]
X_pop = non_raided[features_pop].values
y_pop = non_raided["d_pop"].values
reg_pop = LinearRegression().fit(X_pop, y_pop)
print(f"\nPop model (non-raided):")
print(f"  Intercept: {reg_pop.intercept_:.4f}")
for f, c in zip(features_pop, reg_pop.coef_):
    print(f"  {f}: {c:.4f}")
print(f"  R²: {reg_pop.score(X_pop, y_pop):.4f}")

# Expected pop for raided
raided["logistic"] = raided["food"] * (1 - raided["pop"] / 5.0)
raided["expected_d_pop"] = reg_pop.predict(raided[features_pop].values)
raided["pop_raid_effect"] = raided["d_pop"] - raided["expected_d_pop"]
raided["pop_ratio_raw"] = raided["pop_next"] / raided["pop"].clip(0.01)

print(f"\nRaid pop effect:")
print(f"  Pop delta (actual): mean={raided['d_pop'].mean():.4f}")
print(f"  Pop delta (expected no-raid): mean={raided['expected_d_pop'].mean():.4f}")
print(f"  Raid effect: mean={raided['pop_raid_effect'].mean():.4f}")

# Check: is pop_next / pop ≈ constant for 1-raid cases?
# Look at "light raids" (d_defense between -0.02 and -0.01) vs "heavy" (d_defense < -0.05)
light = raided[(raided["d_defense"] > -0.03) & (raided["d_defense"] < -0.01)]
heavy = raided[raided["d_defense"] < -0.06]

print(f"\nLight raids (d_def in [-0.03, -0.01]): n={len(light)}")
print(f"  Pop ratio: mean={light['pop_ratio_raw'].mean():.4f}, median={light['pop_ratio_raw'].median():.4f}")
print(f"Heavy raids (d_def < -0.06): n={len(heavy)}")
print(f"  Pop ratio: mean={heavy['pop_ratio_raw'].mean():.4f}, median={heavy['pop_ratio_raw'].median():.4f}")

# Pop survival ratio distribution
axes[0, 2].hist(raided["pop_ratio_raw"].clip(0.4, 1.5), bins=80, alpha=0.7, edgecolor="black")
axes[0, 2].axvline(raided["pop_ratio_raw"].median(), color="red", linestyle="--",
                   label=f"Median={raided['pop_ratio_raw'].median():.3f}")
axes[0, 2].set_xlabel("pop_next / pop")
axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title("Pop ratio for raided defenders")
axes[0, 2].legend()

# ================================================================
# 5. Wealth analysis for raided settlements
# ================================================================
print(f"\n--- Wealth ---")
raided["wealth_raid_effect"] = raided["d_wealth"] - non_raided["d_wealth"].mean()

# Is wealth loss proportional to wealth?
axes[0, 3].scatter(raided["wealth"], raided["d_wealth"], alpha=0.1, s=3)
# Fit line
mask_w = raided["wealth"] > 0.01
reg_w = LinearRegression().fit(raided.loc[mask_w, ["wealth"]].values,
                                raided.loc[mask_w, "d_wealth"].values)
x_r = np.linspace(0, 0.5, 100)
axes[0, 3].plot(x_r, reg_w.predict(x_r.reshape(-1, 1)), "r-", linewidth=2,
               label=f"slope={reg_w.coef_[0]:.3f}")
axes[0, 3].set_xlabel("Defender wealth (before)")
axes[0, 3].set_ylabel("Wealth delta")
axes[0, 3].set_title("Defender wealth change vs wealth")
axes[0, 3].legend()

print(f"Wealth delta ~ wealth: slope={reg_w.coef_[0]:.4f}")

# ================================================================
# 6. Takeover mechanics
# ================================================================
raided["def_after"] = raided["defense_next"]
raided["takeover"] = ~raided["owner_same"]

print(f"\n--- Takeover ---")
print(f"Takeover rate: {raided['takeover'].mean():.4f}")

# Takeover probability vs post-raid defense
def_after_bins = np.arange(0, 0.8, 0.02)
raided["def_after_bin"] = pd.cut(raided["def_after"], bins=def_after_bins)
takeover_curve = raided.groupby("def_after_bin", observed=True).agg(
    p=("takeover", "mean"),
    n=("takeover", "count")
).reset_index()

axes[1, 0].bar(range(len(takeover_curve)), takeover_curve["p"], alpha=0.7)
tick_positions = range(0, len(takeover_curve), 5)
axes[1, 0].set_xticks(tick_positions)
axes[1, 0].set_xticklabels([f"{takeover_curve['def_after_bin'].iloc[i].mid:.2f}"
                            for i in tick_positions], rotation=45, fontsize=8)
axes[1, 0].set_xlabel("Defense after raid")
axes[1, 0].set_ylabel("P(takeover)")
axes[1, 0].set_title("Takeover probability vs post-raid defense")

# Defense threshold for takeover
taken = raided[raided["takeover"]]
not_taken = raided[~raided["takeover"]]
print(f"Taken: n={len(taken)}, mean def_after={taken['def_after'].mean():.4f}")
print(f"Not taken: n={len(not_taken)}, mean def_after={not_taken['def_after'].mean():.4f}")
print(f"Max def_after for takeover: {taken['def_after'].max():.4f}")
print(f"99th percentile: {taken['def_after'].quantile(0.99):.4f}")
print(f"95th percentile: {taken['def_after'].quantile(0.95):.4f}")

# ================================================================
# 7. Food changes during raids
# ================================================================
print(f"\n--- Food during raids ---")
print(f"Non-raided d_food: mean={non_raided['d_food'].mean():.4f}, std={non_raided['d_food'].std():.4f}")
print(f"Raided d_food: mean={raided['d_food'].mean():.4f}, std={raided['d_food'].std():.4f}")

axes[1, 1].hist(non_raided["d_food"].clip(-0.3, 0.3), bins=80, alpha=0.5, density=True, label="Non-raided")
axes[1, 1].hist(raided["d_food"].clip(-0.3, 0.3), bins=80, alpha=0.5, density=True, label="Raided")
axes[1, 1].set_xlabel("Food delta")
axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("Food delta: raided vs non-raided")
axes[1, 1].legend()

# ================================================================
# 8. Defense damage vs number of nearby enemies
# ================================================================
# Use n_settlement as proxy for enemy pressure (crude but available)
axes[1, 2].scatter(raided["n_settlement"] + raided["n_port"],
                   raided["d_defense"], alpha=0.1, s=3)
axes[1, 2].set_xlabel("Adjacent settlements+ports")
axes[1, 2].set_ylabel("Defense delta")
axes[1, 2].set_title("Defense damage vs settlement density")

# ================================================================
# 9. Separate single vs multiple raids
# ================================================================
# Hypothesis: defense damage comes in discrete units
# If single raid does ~0.02-0.03 damage, then d_defense in [-0.03, -0.01] ≈ 1 raid
# and d_defense < -0.03 ≈ 2+ raids

single = raided[(raided["d_defense"] > -0.04) & (raided["d_defense"] < -0.01)]
multiple = raided[raided["d_defense"] < -0.06]

print(f"\n--- Single vs multiple raids ---")
print(f"Likely single raid (d_def in [-0.04, -0.01]): n={len(single)}")
print(f"  Pop ratio: median={single['pop_ratio_raw'].median():.4f}")
print(f"  d_wealth: mean={single['d_wealth'].mean():.4f}")
print(f"  d_defense: mean={single['d_defense'].mean():.4f}")

print(f"Likely 2+ raids (d_def < -0.06): n={len(multiple)}")
print(f"  Pop ratio: median={multiple['pop_ratio_raw'].median():.4f}")
print(f"  d_wealth: mean={multiple['d_wealth'].mean():.4f}")
print(f"  d_defense: mean={multiple['d_defense'].mean():.4f}")

# Plot side by side
axes[1, 3].hist(single["d_defense"], bins=40, alpha=0.5, label=f"1 raid (n={len(single)})", density=True)
axes[1, 3].hist(multiple["d_defense"], bins=40, alpha=0.5, label=f"2+ raids (n={len(multiple)})", density=True)
axes[1, 3].set_xlabel("Defense delta")
axes[1, 3].set_ylabel("Density")
axes[1, 3].set_title("Single vs multiple raids")
axes[1, 3].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "raid_analysis_v2.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "raid_analysis_v2.png"), dpi=150)
print(f"\nSaved: {OUT}/raid_analysis_v2.pdf")

# ================================================================
# FINAL SUMMARY
# ================================================================
print(f"\n{'='*60}")
print("FINAL RAID SUMMARY")
print(f"{'='*60}")
print(f"""
1. DEFENSE RECOVERY (non-raided settlements):
   Mean d_defense: {non_raided['d_defense'].mean():.4f} (very small positive)
   Defense recovery is weak and approximately constant

2. RAID DAMAGE (d_defense < -0.01):
   {len(raided)}/{len(alive)} settlements ({len(raided)/len(alive)*100:.1f}%) raided per step
   Mean defense delta: {raided['d_defense'].mean():.4f}
   Median defense delta: {raided['d_defense'].median():.4f}

3. POP SURVIVAL:
   Pop ratio (pop_next/pop) median: {raided['pop_ratio_raw'].median():.4f}
   For light raids: {light['pop_ratio_raw'].median():.4f}
   For heavy raids: {heavy['pop_ratio_raw'].median():.4f}

4. WEALTH:
   Wealth stolen ∝ defender wealth (slope={reg_w.coef_[0]:.3f})
   Mean d_wealth (raided): {raided['d_wealth'].mean():.4f}

5. TAKEOVER:
   Rate: {raided['takeover'].mean()*100:.2f}% of raids
   Tends to happen when post-raid defense < 0.25
   Max post-raid defense at takeover: {taken['def_after'].max():.3f}

6. FOOD:
   Raids do not directly change food
   d_food similar for raided ({raided['d_food'].mean():.4f}) and non-raided ({non_raided['d_food'].mean():.4f})
""")
