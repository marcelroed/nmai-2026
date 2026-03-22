"""
Find the formula for defense recovery as a function of population, food, wealth.

Reproduce:  python3 data-analysis/analyze_defense_recovery.py

Method:
  For each round, collect (def_before, pop_end, food_end, wealth_end, def_delta)
  from settlements where defense increased between steps (no raid).  Fit
  candidate models and report the best per-round fit.

  We use settlements that were alive in BOTH frames, kept the same owner,
  and had defense INCREASE (no raid).  We also exclude steps where
  population jumped unexpectedly (dispersal from winter collapse) or
  dropped (child spawned / raided).

  Key subtlety: defense recovery happens in phase 2.  At that point,
  pop and food have already been updated by phases 1-2.  The values we
  observe at end-of-step ARE the post-phase-2 values (assuming no later
  phase changed them — phases 3-7 don't change pop/food/defense for
  settlements that weren't raided, didn't spawn children, didn't trade,
  and didn't collapse).  So pop_end ≈ pop_at_recovery for clean cases.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np  # type: ignore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_replays_by_round():
    """Return {round_id: [replay, ...]}."""
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


def collect_defense_deltas(replays):
    """Return list of (def_before, def_after, pop_end, food_end, wealth_end, def_delta)
    for clean (unraided, no dispersal, no child spawn) step transitions."""
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
                    continue  # raided or no change

                pop_ratio = sa["population"] / sb["population"] if sb["population"] > 0.01 else 99
                # Filter out dispersal (pop jumped > 20%) and child spawn (pop dropped)
                if pop_ratio > 1.2 or pop_ratio < 0.85:
                    continue

                # Also skip if defense hit the cap (clamped to 1.0)
                if sa["defense"] >= 0.999:
                    continue

                rows.append((
                    sb["defense"],   # def_before
                    sa["defense"],   # def_after
                    sa["population"],  # pop at end of step (≈ pop at phase 2)
                    sa["food"],       # food at end of step
                    sa["wealth"],     # wealth at end of step
                    def_delta,
                ))

    return rows


def try_models(rows, round_id):
    """Try various single-parameter and two-parameter linear models.

    def_delta = c * feature(s)

    where feature is some combination of pop, food, wealth, (1-def), etc.
    """
    arr = np.array(rows, dtype=np.float64)
    def_before = arr[:, 0]
    pop = arr[:, 2]
    food = arr[:, 3]
    wealth = arr[:, 4]
    delta = arr[:, 5]
    one_minus_def = 1.0 - def_before

    # Candidate features (name, vector)
    candidates = [
        ("pop", pop),
        ("food", food),
        ("(1-def)", one_minus_def),
        ("pop*food", pop * food),
        ("pop*(1-def)", pop * one_minus_def),
        ("food*(1-def)", food * one_minus_def),
        ("pop*food*(1-def)", pop * food * one_minus_def),
        ("wealth", wealth),
        ("pop*wealth", pop * wealth),
    ]

    results = []
    for name, feat in candidates:
        if np.all(feat == 0):
            continue
        # Least-squares fit: delta = c * feat
        c = np.dot(feat, delta) / np.dot(feat, feat)
        residuals = delta - c * feat
        mae = np.mean(np.abs(residuals))
        max_err = np.max(np.abs(residuals))
        r2 = 1 - np.sum(residuals ** 2) / np.sum((delta - np.mean(delta)) ** 2)
        results.append((name, c, mae, max_err, r2))

    # Also try two-feature models: delta = a*f1 + b*f2
    two_feat_candidates = [
        ("a*pop + b*(1-def)", np.column_stack([pop, one_minus_def])),
        ("a*pop + b*food", np.column_stack([pop, food])),
        ("a*pop*food + b*(1-def)", np.column_stack([pop * food, one_minus_def])),
        ("a*pop + b*wealth", np.column_stack([pop, wealth])),
        ("a*pop + b*pop*food", np.column_stack([pop, pop * food])),
    ]
    for name, X in two_feat_candidates:
        coeffs, res, _, _ = np.linalg.lstsq(X, delta, rcond=None)
        pred = X @ coeffs
        residuals = delta - pred
        mae = np.mean(np.abs(residuals))
        max_err = np.max(np.abs(residuals))
        r2 = 1 - np.sum(residuals ** 2) / np.sum((delta - np.mean(delta)) ** 2)
        c_str = ", ".join(f"{c:.6f}" for c in coeffs)
        results.append((f"{name} [{c_str}]", coeffs[0], mae, max_err, r2))

    return sorted(results, key=lambda x: x[2])  # sort by MAE


def main():
    rounds = load_replays_by_round()
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays from "
          f"{len(rounds)} rounds\n")

    all_round_results = {}

    for rid in sorted(rounds):
        replays = rounds[rid]
        rows = collect_defense_deltas(replays)
        if len(rows) < 50:
            print(f"Round {rid[:12]}: only {len(rows)} clean rows, skipping")
            continue

        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}  ({len(rows)} clean defense-increase observations)")
        print(f"{'=' * 72}")

        results = try_models(rows, rid)

        print(f"\n{'Model':>45s}  {'coeff':>10s}  {'MAE':>10s}  {'max_err':>10s}  {'R²':>8s}")
        print("-" * 90)
        for name, c, mae, max_err, r2 in results[:12]:
            c_str = f"{c:.6f}" if isinstance(c, float) else f"{c:.6f}"
            print(f"{name:>45s}  {c_str:>10s}  {mae:>10.6f}  {max_err:>10.6f}  {r2:>8.4f}")

        best_name = results[0][0]
        best_mae = results[0][2]
        all_round_results[rid] = results

        print(f"\n  Best single model: {best_name} (MAE={best_mae:.6f})")
        print()

    # ── Cross-round comparison ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CROSS-ROUND COMPARISON: best model type per round")
    print("=" * 72)

    # For each model type, show coefficient and MAE across rounds
    model_names_to_check = ["pop", "food", "(1-def)", "pop*food", "pop*(1-def)"]

    print(f"\n{'Round':>12s}", end="")
    for mn in model_names_to_check:
        print(f"  {mn:>16s}", end="")
    print()
    print("-" * (14 + 18 * len(model_names_to_check)))

    for rid in sorted(all_round_results):
        results = all_round_results[rid]
        result_dict = {r[0].split(" [")[0]: r for r in results}
        print(f"{rid[:12]:>12s}", end="")
        for mn in model_names_to_check:
            r = result_dict.get(mn)
            if r:
                _, c, mae, _, r2 = r
                print(f"  c={c:.4f} R²={r2:.3f}", end="")
            else:
                print(f"  {'N/A':>16s}", end="")
        print()

    # ── Detailed best-model analysis ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("DETAILED: coefficient stability within each round")
    print("=" * 72)

    # For the best model type, check if the coefficient is stable
    # within a round by splitting data into early vs late steps
    for rid in sorted(all_round_results):
        replays = rounds[rid]
        rows = collect_defense_deltas(replays)
        if len(rows) < 100:
            continue

        arr = np.array(rows, dtype=np.float64)
        pop = arr[:, 2]
        food = arr[:, 3]
        delta = arr[:, 5]

        # Check top models with per-seed split
        for feat_name, feat in [("pop", pop), ("pop*food", pop * food)]:
            c_global = np.dot(feat, delta) / np.dot(feat, feat)
            residuals = delta - c_global * feat
            mae = np.mean(np.abs(residuals))

            # Split by replay seed to check consistency
            seed_rows = defaultdict(list)
            idx = 0
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
                        dd = sa["defense"] - sb["defense"]
                        if dd <= 0:
                            continue
                        pr = sa["population"] / sb["population"] if sb["population"] > 0.01 else 99
                        if pr > 1.2 or pr < 0.85:
                            continue
                        if sa["defense"] >= 0.999:
                            continue

                        seed_rows[replay["seed_index"]].append(idx)
                        idx += 1

            if feat_name == "pop":
                print(f"\n  Round {rid[:12]} — model: delta = c * {feat_name} (c={c_global:.6f}, MAE={mae:.6f})")
                for seed in sorted(seed_rows):
                    idxs = seed_rows[seed]
                    f_s = feat[idxs]
                    d_s = delta[idxs]
                    c_s = np.dot(f_s, d_s) / np.dot(f_s, f_s)
                    mae_s = np.mean(np.abs(d_s - c_s * f_s))
                    print(f"    seed={seed}: c={c_s:.6f}, MAE={mae_s:.6f}, n={len(idxs)}")


if __name__ == "__main__":
    main()
