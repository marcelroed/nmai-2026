"""
Q9: What things are still modeled the worst from one step to the next?
Can you find a model that is better?

Build models for food, pop, defense, wealth. Compare residuals.
Focus on non-raid transitions to isolate deterministic mechanics.
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

replays = load_all_replays()
df = build_transition_df(replays)

# Non-raided, alive, same owner
alive = df[(df["alive"]) & (df["alive_next"])].copy()
alive["owner_same"] = alive["owner_id"] == alive["owner_id_next"]
clean = alive[(alive["d_defense"] >= -0.001) & alive["owner_same"]].copy()

# Feature engineering
clean["food_x_1mfood"] = clean["food"] * (1 - clean["food"])
clean["n_settl_adj"] = clean["n_settlement"] + clean["n_port"] + clean["n_ruin"]
clean["is_port"] = clean["has_port"].astype(int)
clean["plains_x_1mf"] = clean["n_plains"] * (1 - clean["food"])
clean["forest_x_1mf"] = clean["n_forest"] * (1 - clean["food"])
clean["mount_x_1mf"] = clean["n_mountain"] * (1 - clean["food"])
clean["settl_x_1mf"] = clean["n_settl_adj"] * (1 - clean["food"])
clean["logistic"] = clean["food"] * (1 - clean["pop"] / 5.0)
clean["pop_x_food"] = clean["pop"] * clean["food"]

print(f"Clean (non-raided, alive, same-owner) transitions: {len(clean)}")

# ================================================================
# Build best model for each variable
# ================================================================
results = {}

# --- FOOD ---
food_feats = ["pop", "food", "food_x_1mfood", "plains_x_1mf", "forest_x_1mf",
              "mount_x_1mf", "settl_x_1mf", "is_port"]
X_food = clean[food_feats].values
y_food = clean["d_food"].values
reg_food = LinearRegression().fit(X_food, y_food)
food_r2 = reg_food.score(X_food, y_food)
clean["food_pred"] = reg_food.predict(X_food)
clean["food_resid"] = y_food - clean["food_pred"]
results["food"] = {"R2": food_r2, "MAE": np.abs(clean["food_resid"]).mean(),
                   "RMSE": np.sqrt(np.mean(clean["food_resid"]**2))}

# --- POPULATION ---
# Simple model: logistic growth
pop_feats_v1 = ["logistic"]
X_pop1 = clean[pop_feats_v1].values
y_pop = clean["d_pop"].values
reg_pop1 = LinearRegression().fit(X_pop1, y_pop)
pop_r2_v1 = reg_pop1.score(X_pop1, y_pop)

# Better model: include pop, food, interactions
pop_feats_v2 = ["pop", "food", "logistic", "pop_x_food", "is_port"]
X_pop2 = clean[pop_feats_v2].values
reg_pop2 = LinearRegression().fit(X_pop2, y_pop)
pop_r2_v2 = reg_pop2.score(X_pop2, y_pop)

# Even better: add spawn cost proxy (high pop+food = likely spawner)
clean["spawn_proxy"] = clean["pop"] * clean["food"] * (clean["pop"] > 1.0).astype(float)
pop_feats_v3 = ["pop", "food", "logistic", "pop_x_food", "is_port", "spawn_proxy"]
X_pop3 = clean[pop_feats_v3].values
reg_pop3 = LinearRegression().fit(X_pop3, y_pop)
pop_r2_v3 = reg_pop3.score(X_pop3, y_pop)

clean["pop_pred"] = reg_pop3.predict(X_pop3)
clean["pop_resid"] = y_pop - clean["pop_pred"]
results["pop"] = {"R2": pop_r2_v3, "MAE": np.abs(clean["pop_resid"]).mean(),
                  "RMSE": np.sqrt(np.mean(clean["pop_resid"]**2))}

print(f"\n--- Pop models ---")
print(f"  v1 (logistic only): R²={pop_r2_v1:.4f}")
print(f"  v2 (+pop, food, interactions): R²={pop_r2_v2:.4f}")
print(f"  v3 (+spawn_proxy): R²={pop_r2_v3:.4f}")

# --- DEFENSE ---
def_feats_v1 = ["defense"]
X_def1 = clean[def_feats_v1].values
y_def = clean["d_defense"].values
reg_def1 = LinearRegression().fit(X_def1, y_def)
def_r2_v1 = reg_def1.score(X_def1, y_def)

# Add more features
def_feats_v2 = ["defense", "pop", "food", "is_port"]
X_def2 = clean[def_feats_v2].values
reg_def2 = LinearRegression().fit(X_def2, y_def)
def_r2_v2 = reg_def2.score(X_def2, y_def)

# Quadratic in defense
clean["defense_sq"] = clean["defense"] ** 2
clean["def_x_1mdef"] = clean["defense"] * (1 - clean["defense"])
def_feats_v3 = ["defense", "defense_sq", "pop", "food", "is_port"]
X_def3 = clean[def_feats_v3].values
reg_def3 = LinearRegression().fit(X_def3, y_def)
def_r2_v3 = reg_def3.score(X_def3, y_def)

clean["def_pred"] = reg_def3.predict(X_def3)
clean["def_resid"] = y_def - clean["def_pred"]
results["defense"] = {"R2": def_r2_v3, "MAE": np.abs(clean["def_resid"]).mean(),
                      "RMSE": np.sqrt(np.mean(clean["def_resid"]**2))}

print(f"\n--- Defense models ---")
print(f"  v1 (defense only): R²={def_r2_v1:.4f}")
print(f"  v2 (+pop, food, port): R²={def_r2_v2:.4f}")
print(f"  v3 (+defense²): R²={def_r2_v3:.4f}")
print(f"  Coefs: {dict(zip(def_feats_v3, reg_def3.coef_))}")

# --- WEALTH ---
wealth_feats_v1 = ["wealth"]
X_w1 = clean[wealth_feats_v1].values
y_w = clean["d_wealth"].values
reg_w1 = LinearRegression().fit(X_w1, y_w)
wealth_r2_v1 = reg_w1.score(X_w1, y_w)

wealth_feats_v2 = ["wealth", "pop", "food", "is_port", "defense"]
X_w2 = clean[wealth_feats_v2].values
reg_w2 = LinearRegression().fit(X_w2, y_w)
wealth_r2_v2 = reg_w2.score(X_w2, y_w)

# Wealth might depend on trade (port, neighbors)
clean["wealth_sq"] = clean["wealth"] ** 2
wealth_feats_v3 = ["wealth", "wealth_sq", "pop", "food", "is_port", "defense",
                   "n_port"]
X_w3 = clean[wealth_feats_v3].values
reg_w3 = LinearRegression().fit(X_w3, y_w)
wealth_r2_v3 = reg_w3.score(X_w3, y_w)

clean["wealth_pred"] = reg_w3.predict(X_w3)
clean["wealth_resid"] = y_w - clean["wealth_pred"]
results["wealth"] = {"R2": wealth_r2_v3, "MAE": np.abs(clean["wealth_resid"]).mean(),
                     "RMSE": np.sqrt(np.mean(clean["wealth_resid"]**2))}

print(f"\n--- Wealth models ---")
print(f"  v1 (wealth only): R²={wealth_r2_v1:.4f}")
print(f"  v2 (+pop, food, port, defense): R²={wealth_r2_v2:.4f}")
print(f"  v3 (+wealth², n_port): R²={wealth_r2_v3:.4f}")
print(f"  Coefs: {dict(zip(wealth_feats_v3, reg_w3.coef_))}")

# ================================================================
# Summary comparison
# ================================================================
print(f"\n{'='*60}")
print("MODEL COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Variable':<12} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
for var, metrics in results.items():
    print(f"{var:<12} {metrics['R2']:>8.4f} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f}")

# ================================================================
# Try gradient boosting for worst-performing variable
# ================================================================
worst_var = min(results, key=lambda x: results[x]["R2"])
print(f"\nWorst modeled variable: {worst_var} (R²={results[worst_var]['R2']:.4f})")

# Try GBM on the worst variable
all_feats = ["pop", "food", "wealth", "defense", "is_port",
             "n_plains", "n_forest", "n_mountain", "n_ocean",
             "n_settlement", "n_port", "n_ruin"]

if worst_var == "food":
    y_target = clean["d_food"].values
elif worst_var == "pop":
    y_target = clean["d_pop"].values
elif worst_var == "defense":
    y_target = clean["d_defense"].values
else:
    y_target = clean["d_wealth"].values

X_all = clean[all_feats].values

# Quick cross-validation with GBM
print(f"\nGradient Boosting for {worst_var}:")
gbm = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42,
                                 subsample=0.8, learning_rate=0.1)
cv_scores = cross_val_score(gbm, X_all, y_target, cv=5, scoring="r2")
print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Fit GBM and get feature importance
gbm.fit(X_all, y_target)
print(f"  Train R²: {gbm.score(X_all, y_target):.4f}")
print(f"\n  Feature importance:")
for name, imp in sorted(zip(all_feats, gbm.feature_importances_), key=lambda x: -x[1]):
    print(f"    {name}: {imp:.4f}")

# ================================================================
# ALSO try GBM on all variables for comparison
# ================================================================
print(f"\n{'='*60}")
print("GBM RESULTS FOR ALL VARIABLES")
print(f"{'='*60}")
gbm_results = {}
for var, y_col in [("food", "d_food"), ("pop", "d_pop"),
                   ("defense", "d_defense"), ("wealth", "d_wealth")]:
    y_t = clean[y_col].values
    gbm_t = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42,
                                       subsample=0.8, learning_rate=0.1)
    cv = cross_val_score(gbm_t, X_all, y_t, cv=5, scoring="r2")
    gbm_t.fit(X_all, y_t)
    gbm_results[var] = {"CV_R2": cv.mean(), "Train_R2": gbm_t.score(X_all, y_t)}
    print(f"  {var}: CV R²={cv.mean():.4f}, Train R²={gbm_t.score(X_all, y_t):.4f}")

# ================================================================
# Detailed residual analysis for worst variable
# ================================================================
# Use food as example since we have the best model
# Analyze: where does the food model fail?
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, (var, y_col, pred_col, resid_col) in enumerate([
    ("food", "d_food", "food_pred", "food_resid"),
    ("pop", "d_pop", "pop_pred", "pop_resid"),
    ("defense", "d_defense", "def_pred", "def_resid"),
    ("wealth", "d_wealth", "wealth_pred", "wealth_resid"),
]):
    # Predicted vs actual
    sample = clean.sample(min(5000, len(clean)), random_state=42)
    axes[0, i].scatter(sample[pred_col], sample[y_col], alpha=0.1, s=3)
    lims = [min(sample[y_col].min(), sample[pred_col].min()),
            max(sample[y_col].max(), sample[pred_col].max())]
    axes[0, i].plot(lims, lims, "r--", linewidth=1)
    axes[0, i].set_xlabel("Predicted")
    axes[0, i].set_ylabel("Actual")
    axes[0, i].set_title(f"{var}: R²={results[var]['R2']:.3f}")

    # Residual histogram
    axes[1, i].hist(clean[resid_col].clip(-0.3, 0.3), bins=80, alpha=0.7, edgecolor="black")
    axes[1, i].set_xlabel("Residual")
    axes[1, i].set_ylabel("Count")
    axes[1, i].set_title(f"{var}: MAE={results[var]['MAE']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "worst_modeled.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "worst_modeled.png"), dpi=150)
print(f"\nSaved: {OUT}/worst_modeled.pdf")

# ================================================================
# Deep dive into the worst variable: what causes large residuals?
# ================================================================
print(f"\n{'='*60}")
print(f"DEEP DIVE: {worst_var} residuals")
print(f"{'='*60}")

resid_col = f"{worst_var}_resid" if worst_var != "defense" else "def_resid"
large_resid = clean[np.abs(clean[resid_col]) > clean[resid_col].std() * 2]
print(f"Large residuals (|resid| > 2σ): {len(large_resid)} ({len(large_resid)/len(clean)*100:.1f}%)")

# Compare large-residual cases to normal cases
normal = clean[np.abs(clean[resid_col]) <= clean[resid_col].std() * 2]
for feat in ["pop", "food", "wealth", "defense", "is_port", "step",
             "n_settlement", "n_port", "n_ruin"]:
    if feat in clean.columns:
        print(f"  {feat}: normal={normal[feat].mean():.3f}, large_resid={large_resid[feat].mean():.3f}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print(f"\n{'='*60}")
print("FINAL FINDINGS")
print(f"{'='*60}")
print(f"""
LINEAR MODEL PERFORMANCE (non-raided settlements):
  Food:    R²={results['food']['R2']:.4f}  MAE={results['food']['MAE']:.4f}  RMSE={results['food']['RMSE']:.4f}
  Pop:     R²={results['pop']['R2']:.4f}  MAE={results['pop']['MAE']:.4f}  RMSE={results['pop']['RMSE']:.4f}
  Defense: R²={results['defense']['R2']:.4f}  MAE={results['defense']['MAE']:.4f}  RMSE={results['defense']['RMSE']:.4f}
  Wealth:  R²={results['wealth']['R2']:.4f}  MAE={results['wealth']['MAE']:.4f}  RMSE={results['wealth']['RMSE']:.4f}

WORST MODELED: {worst_var}

GBM IMPROVEMENT:
""")
for var in results:
    linear = results[var]["R2"]
    gbm_cv = gbm_results[var]["CV_R2"]
    print(f"  {var}: Linear R²={linear:.4f} → GBM CV R²={gbm_cv:.4f} (Δ={gbm_cv-linear:+.4f})")
