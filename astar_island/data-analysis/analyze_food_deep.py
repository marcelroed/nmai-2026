"""
Deep food analysis: isolate food production, winter severity, and weather.

Strategy:
1. Look at all steps, but filter out raids properly
2. Account for trade effects (port settlements)
3. Account for clamping (food=0, food=0.998)
4. Fit food model per-round to check stability
5. Extract winter severity from newborn settlements
6. Compare food model variants
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_replays import load_all_replays, build_transition_df, build_spawn_df
from sklearn.linear_model import LinearRegression

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

replays = load_all_replays()
df = build_transition_df(replays)
spawns = build_spawn_df(replays)

# ================================================================
# Better filtering: exclude raids using a threshold
# Natural defense recovery: |d_def| < 0.015 typically
# Raids cause d_def < -0.01 at minimum
# Use threshold: d_defense >= -0.005 to exclude raids
# Also exclude ownership changes and deaths
# ================================================================
mask = (df["alive"]) & (df["alive_next"]) & (df["d_defense"] >= -0.005)
mask &= (df["owner_id"] == df["owner_id_next"])
# Also exclude clamped cases (food at 0 or near cap)
mask &= (df["food"] > 0.01) & (df["food"] < 0.99)
mask &= (df["food_next"] > 0.001) & (df["food_next"] < 0.999)
clean = df[mask].copy()
print(f"Clean transitions (no raids, no clamp): {len(clean)}")

# ================================================================
# Feature engineering
# ================================================================
clean["food_x_1mfood"] = clean["food"] * (1 - clean["food"])
clean["n_settl_adj"] = clean["n_settlement"] + clean["n_port"] + clean["n_ruin"]
clean["is_port"] = clean["has_port"].astype(int)

# ================================================================
# Model comparison: try several functional forms
# ================================================================
models = {}

# Model A: basic linear (current)
feats_a = ["pop", "food_x_1mfood", "n_plains", "n_forest", "n_mountain", "n_settl_adj"]
X_a = clean[feats_a].values
y = clean["d_food"].values
reg_a = LinearRegression().fit(X_a, y)
models["A: basic"] = (reg_a, feats_a, reg_a.score(X_a, y))

# Model B: add food linearly (separate food and food*(1-food))
feats_b = ["pop", "food", "food_x_1mfood", "n_plains", "n_forest", "n_mountain", "n_settl_adj"]
X_b = clean[feats_b].values
reg_b = LinearRegression().fit(X_b, y)
models["B: +food"] = (reg_b, feats_b, reg_b.score(X_b, y))

# Model C: add port status
feats_c = ["pop", "food_x_1mfood", "n_plains", "n_forest", "n_mountain", "n_settl_adj", "is_port"]
X_c = clean[feats_c].values
reg_c = LinearRegression().fit(X_c, y)
models["C: +port"] = (reg_c, feats_c, reg_c.score(X_c, y))

# Model D: terrain interacted with (1-food)
clean["plains_x_1mf"] = clean["n_plains"] * (1 - clean["food"])
clean["forest_x_1mf"] = clean["n_forest"] * (1 - clean["food"])
clean["mount_x_1mf"] = clean["n_mountain"] * (1 - clean["food"])
clean["settl_x_1mf"] = clean["n_settl_adj"] * (1 - clean["food"])
feats_d = ["pop", "food_x_1mfood", "plains_x_1mf", "forest_x_1mf", "mount_x_1mf", "settl_x_1mf"]
X_d = clean[feats_d].values
reg_d = LinearRegression().fit(X_d, y)
models["D: terrain×(1-f)"] = (reg_d, feats_d, reg_d.score(X_d, y))

# Model E: D + port
feats_e = feats_d + ["is_port"]
X_e = clean[feats_e].values
reg_e = LinearRegression().fit(X_e, y)
models["E: D+port"] = (reg_e, feats_e, reg_e.score(X_e, y))

# Model F: all interactions
clean["pop_x_food"] = clean["pop"] * clean["food"]
feats_f = ["pop", "food", "food_x_1mfood", "plains_x_1mf", "forest_x_1mf", "mount_x_1mf", "settl_x_1mf", "is_port"]
X_f = clean[feats_f].values
reg_f = LinearRegression().fit(X_f, y)
models["F: full"] = (reg_f, feats_f, reg_f.score(X_f, y))

print("\n--- Model comparison ---")
for name, (reg, feats, r2) in models.items():
    print(f"{name}: R²={r2:.4f}")
    print(f"  Intercept: {reg.intercept_:.6f}")
    for fname, coef in zip(feats, reg.coef_):
        print(f"  {fname}: {coef:+.6f}")
    print()

# ================================================================
# Use best model (F) with fixed effects to extract weather
# ================================================================
best_name = "F: full"
best_reg, best_feats, best_r2 = models[best_name]

# Fixed effects by (round, seed, step)
group_means = clean.groupby(["round_id", "seed", "step"])[best_feats + ["d_food"]].transform("mean")
X_dm = clean[best_feats].values - group_means[best_feats].values
y_dm = clean["d_food"].values - group_means["d_food"].values

reg_fe = LinearRegression(fit_intercept=False).fit(X_dm, y_dm)
print(f"\n--- Fixed-effects model ({best_name}) ---")
for fname, coef in zip(best_feats, reg_fe.coef_):
    print(f"  {fname}: {coef:+.6f}")
within_r2 = reg_fe.score(X_dm, y_dm)
print(f"R² (within): {within_r2:.4f}")

# Extract weather
clean["food_pred_fe"] = clean[best_feats].values @ reg_fe.coef_
clean["food_resid_fe"] = clean["d_food"] - clean["food_pred_fe"]

weather = clean.groupby(["round_id", "seed", "step"]).agg(
    weather=("food_resid_fe", "mean"),
    resid_std=("food_resid_fe", "std"),
    n=("food_resid_fe", "count"),
    mean_food=("food", "mean"),
    mean_pop=("pop", "mean"),
).reset_index()

print(f"\n--- Weather from FE model ---")
print(f"Mean: {weather['weather'].mean():.6f}")
print(f"Std: {weather['weather'].std():.6f}")
print(f"Range: [{weather['weather'].min():.4f}, {weather['weather'].max():.4f}]")
print(f"Mean within-group residual std: {weather['resid_std'].mean():.6f}")

# ================================================================
# WINTER SEVERITY: Extract from newborn settlements
# ================================================================
print(f"\n{'='*60}")
print("WINTER SEVERITY from newborn settlements")
print(f"{'='*60}")

# Newborn settlements: created during growth phase, then only experience
# conflict, trade, winter. Their food at observation = initial_food - winter + noise
# For ruin spawns with no adjacent enemies and no port: food = initial_food - winter

# Filter: ruin spawns (cleanest signal), not raided (defense = 0.15 still)
ruin_spawns = spawns[spawns["terrain_before"] == 3].copy()
print(f"Ruin spawns: {len(ruin_spawns)}")
print(f"  child_pop: {ruin_spawns['child_pop'].unique()}")  # should be [0.4]
print(f"  child_defense: {ruin_spawns['child_defense'].unique()}")  # should be [0.15]

# Non-raided ruin spawns (defense still 0.15)
clean_ruin = ruin_spawns[ruin_spawns["child_defense"] == 0.15].copy()
print(f"Clean ruin spawns (defense=0.15): {len(clean_ruin)}")
print(f"  child_food: mean={clean_ruin['child_food'].mean():.4f}, std={clean_ruin['child_food'].std():.4f}")

# If initial_food_ruin = X, then child_food = X - winter_severity + weather
# Group by step to see if there's a pattern
ruin_food_by_step = clean_ruin.groupby(["round_id", "seed", "step"])["child_food"].agg(["mean", "std", "count"])
print(f"\nRuin spawn food by (round, seed, step):")
print(f"  Mean of group means: {ruin_food_by_step['mean'].mean():.4f}")
print(f"  Std of group means: {ruin_food_by_step['mean'].std():.4f}")

# Similarly for plains/forest spawns
plains_spawns = spawns[spawns["terrain_before"] == 11].copy()
clean_plains = plains_spawns[plains_spawns["child_defense"] == 0.2].copy()
print(f"\nClean plains spawns (defense=0.2): {len(clean_plains)}")
print(f"  child_food: mean={clean_plains['child_food'].mean():.4f}, std={clean_plains['child_food'].std():.4f}")

forest_spawns = spawns[spawns["terrain_before"] == 4].copy()
clean_forest = forest_spawns[forest_spawns["child_defense"] == 0.2].copy()
print(f"Clean forest spawns (defense=0.2): {len(clean_forest)}")
print(f"  child_food: mean={clean_forest['child_food'].mean():.4f}, std={clean_forest['child_food'].std():.4f}")

# ================================================================
# Cross-check: do child food values match the weather per step?
# If child_food = initial_food + food_production_child - winter + weather,
# and we know weather from the FE model, then:
# child_food - weather = initial_food + food_production_child - winter (constant)
# ================================================================
# Merge weather into ruin spawns
clean_ruin_w = clean_ruin.merge(
    weather[["round_id", "seed", "step", "weather"]],
    on=["round_id", "seed", "step"],
    how="left"
)
clean_ruin_w = clean_ruin_w[clean_ruin_w["weather"].notna()]
clean_ruin_w["food_minus_weather"] = clean_ruin_w["child_food"] - clean_ruin_w["weather"]

print(f"\nRuin child_food - weather:")
print(f"  Mean: {clean_ruin_w['food_minus_weather'].mean():.4f}")
print(f"  Std: {clean_ruin_w['food_minus_weather'].std():.4f}")
print(f"  (Should be constant if weather explains the variation)")

# Also check: child_food vs weather correlation
corr = clean_ruin_w[["child_food", "weather"]].corr().iloc[0, 1]
print(f"  Correlation(child_food, weather): {corr:.4f}")

# ================================================================
# PLOTS
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. d_food vs food - raw scatter
sample = clean.sample(min(10000, len(clean)), random_state=42)
axes[0, 0].scatter(sample["food"], sample["d_food"], alpha=0.1, s=3)
# Overlay model prediction (using OLS model F)
food_range = np.linspace(0.01, 0.99, 100)
# For "typical" settlement: pop=2, n_plains=3, n_forest=1, n_mountain=0, n_settl_adj=1, port=0
typical = np.zeros((100, len(best_feats)))
for i, f in enumerate(food_range):
    typical[i] = [2.0, f, f*(1-f), 3*(1-f), 1*(1-f), 0, 1*(1-f), 0]
pred = best_reg.predict(typical)
axes[0, 0].plot(food_range, pred, "r-", linewidth=2, label="Model F (typical)")
axes[0, 0].set_xlabel("Food")
axes[0, 0].set_ylabel("d_food")
axes[0, 0].set_title("Food delta vs current food")
axes[0, 0].legend()

# 2. Residuals by port status
for port, label, color in [(False, "No Port", "blue"), (True, "Port", "orange")]:
    sub = clean[clean["has_port"] == port]
    pred = best_reg.predict(sub[best_feats].values)
    resid = sub["d_food"].values - pred
    axes[0, 1].hist(resid, bins=80, alpha=0.5, label=f"{label} (n={len(sub)})", density=True, color=color)
axes[0, 1].set_xlabel("Residual")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title("Food model residuals by port status")
axes[0, 1].legend()

# 3. Weather distribution
axes[0, 2].hist(weather["weather"], bins=60, edgecolor="black", alpha=0.7)
axes[0, 2].axvline(weather["weather"].mean(), color="red", linestyle="--",
                   label=f"Mean={weather['weather'].mean():.4f}")
axes[0, 2].set_xlabel("Weather component")
axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title("Weather distribution (per round/seed/step)")
axes[0, 2].legend()

# 4. Child food histogram with weather overlay
axes[1, 0].hist(clean_ruin["child_food"], bins=40, alpha=0.5, label="Ruin spawn", density=True)
axes[1, 0].hist(clean_plains["child_food"], bins=40, alpha=0.5, label="Plains spawn", density=True)
axes[1, 0].set_xlabel("Child food at birth")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title("Newborn food distribution by terrain")
axes[1, 0].legend()

# 5. Child food vs weather (should correlate if winter is constant)
if len(clean_ruin_w) > 10:
    axes[1, 1].scatter(clean_ruin_w["weather"], clean_ruin_w["child_food"], alpha=0.3, s=10)
    axes[1, 1].set_xlabel("Weather (from FE model)")
    axes[1, 1].set_ylabel("Ruin child food")
    axes[1, 1].set_title(f"Ruin child food vs weather (r={corr:.3f})")
    # Fit line
    z = np.polyfit(clean_ruin_w["weather"], clean_ruin_w["child_food"], 1)
    x_line = np.linspace(clean_ruin_w["weather"].min(), clean_ruin_w["weather"].max(), 100)
    axes[1, 1].plot(x_line, z[0]*x_line + z[1], "r-", linewidth=2)

# 6. Model R² comparison
model_names = list(models.keys())
model_r2s = [models[n][2] for n in model_names]
axes[1, 2].barh(model_names, model_r2s, alpha=0.7)
axes[1, 2].set_xlabel("R²")
axes[1, 2].set_title("Food model comparison")
axes[1, 2].set_xlim(0.45, max(model_r2s) + 0.05)
for i, v in enumerate(model_r2s):
    axes[1, 2].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "food_deep_analysis.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "food_deep_analysis.png"), dpi=150)
print(f"\nSaved: {OUT}/food_deep_analysis.pdf")

# ================================================================
# Per-round coefficient stability
# ================================================================
print(f"\n{'='*60}")
print("Per-round food coefficient stability")
print(f"{'='*60}")
coefs_by_round = []
for rid, g in clean.groupby("round_id"):
    if len(g) < 100:
        continue
    X_r = g[best_feats].values
    y_r = g["d_food"].values
    reg_r = LinearRegression().fit(X_r, y_r)
    row = {"round_id": rid, "intercept": reg_r.intercept_, "R2": reg_r.score(X_r, y_r)}
    for fname, coef in zip(best_feats, reg_r.coef_):
        row[fname] = coef
    coefs_by_round.append(row)

coefs_df = pd.DataFrame(coefs_by_round)
print(coefs_df.to_string(index=False))

print(f"\nCoefficient variability (CV = std/|mean|):")
for col in ["intercept"] + best_feats:
    m = coefs_df[col].mean()
    s = coefs_df[col].std()
    cv = s / abs(m) if abs(m) > 1e-6 else float('inf')
    print(f"  {col}: mean={m:+.4f}, std={s:.4f}, CV={cv:.2f}")

# ================================================================
# CONCLUSION
# ================================================================
print(f"\n{'='*60}")
print("CONCLUSIONS")
print(f"{'='*60}")
print(f"""
1. WINTER SEVERITY:
   The weather component (food intercept + stochastic weather - winter) has:
   - Mean: {weather['weather'].mean():.4f}
   - Std: {weather['weather'].std():.4f}
   This represents: food_base - winter_severity + stochastic_weather

2. From newborn ruin settlements (unraided):
   - Mean child food: {clean_ruin['child_food'].mean():.4f}
   - This = initial_food_ruin + food_prod_ruin - winter + weather
   - food_minus_weather mean: {clean_ruin_w['food_minus_weather'].mean():.4f}
   - If initial_food_ruin ≈ 0.2 (typical), then winter ≈ 0.2 - {clean_ruin_w['food_minus_weather'].mean():.4f} ≈ {0.2 - clean_ruin_w['food_minus_weather'].mean():.4f}

3. FOOD MODEL (best: {best_name}, R²={best_r2:.4f}):
   Intercept: {best_reg.intercept_:+.6f}
""")
for fname, coef in zip(best_feats, best_reg.coef_):
    print(f"   {fname}: {coef:+.6f}")
