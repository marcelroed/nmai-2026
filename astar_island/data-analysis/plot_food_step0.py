"""
Step 0 analysis: At step 0, all settlements have food=0.5, pop=0.5.
Within a single seed, weather is constant. So food_delta should be a
PURE function of terrain.

This isolates the terrain coefficients with no confounders.

Reproduce:  python3 data-analysis/plot_food_step0.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


def terrain_adj_full(grid, x, y):
    h, w = len(grid), len(grid[0])
    counts = defaultdict(int)
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if nx == x and ny == y:
                continue
            counts[grid[ny][nx]] += 1
    return dict(counts)


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


def fit_ols(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    res = y - pred
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def main():
    rounds = load_replays_by_round()

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: For each round, each seed, step 0→1 scatter
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]

        for replay in replays:
            fb, fa = replay["frames"][0], replay["frames"][1]
            grid = fb["grid"]
            seed = replay["seed_index"]

            before_map = {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}
            after_map = {(s["x"], s["y"]): s for s in fa["settlements"] if s["alive"]}

            foods_b, foods_d = [], []
            pops_b = []
            n_pls, n_fos, n_mts, n_ocs, n_sts = [], [], [], [], []
            xs, ys = [], []

            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                if grid[y][x] == PORT:
                    continue
                adj = terrain_adj_full(grid, x, y)

                foods_b.append(sb["food"])
                foods_d.append(sa["food"] - sb["food"])
                pops_b.append(sb["population"])
                n_pls.append(adj.get(PLAINS, 0))
                n_fos.append(adj.get(FOREST, 0))
                n_mts.append(adj.get(MOUNTAIN, 0))
                n_ocs.append(adj.get(OCEAN, 0))
                n_sts.append(adj.get(SETTLEMENT, 0) + adj.get(PORT, 0) + adj.get(RUIN, 0))

            if not foods_d:
                continue

            foods_b = np.array(foods_b)
            foods_d = np.array(foods_d)
            pops_b = np.array(pops_b)
            n_pls = np.array(n_pls)
            n_fos = np.array(n_fos)
            n_mts = np.array(n_mts)
            n_ocs = np.array(n_ocs)
            n_sts = np.array(n_sts)

            # Verify food and pop are constant
            if foods_b.std() > 0.001:
                print(f"  WARNING: food_b std={foods_b.std():.6f} (not constant!)")

            terrain_score = n_pls + 1.5 * n_fos

            ax.scatter(terrain_score, foods_d, s=15, alpha=0.6,
                      label=f's={seed}' if seed < 3 else None)

            if seed == 0:
                # Fit line on this seed
                X = np.column_stack([np.ones(len(foods_d)), n_pls, n_fos, n_sts])
                c, r2, mae, res = fit_ols(X, foods_d)
                print(f"Round {rid[:12]} seed={seed}: n={len(foods_d)}, "
                      f"food_b={foods_b[0]:.3f}, pop_b={pops_b[0]:.3f}")
                print(f"  fit: int={c[0]:.6f}, pl={c[1]:.6f}, fo={c[2]:.6f}, st={c[3]:.6f}")
                print(f"  R²={r2:.6f}, MAE={mae:.6f}")
                print(f"  max|res|={np.max(np.abs(res)):.6f}")

                # Fit without n_st
                X2 = np.column_stack([np.ones(len(foods_d)), n_pls, n_fos])
                c2, r2_2, mae2, res2 = fit_ols(X2, foods_d)
                print(f"  without n_st: R²={r2_2:.6f}, MAE={mae2:.6f}")

                # Check residuals by n_st
                for sv in sorted(set(n_sts)):
                    sv_mask = n_sts == sv
                    if sv_mask.sum() >= 3:
                        print(f"    n_st={int(sv)}: n={sv_mask.sum()}, "
                              f"mean_res={res2[sv_mask].mean():.6f}, "
                              f"std_res={res2[sv_mask].std():.6f}")

        ax.set_xlabel('terrain_score (pl + 1.5×fo)')
        ax.set_ylabel('food_delta (step 0→1)')
        ax.set_title(f'Round {rid[:8]}')
        ax.legend(fontsize=6, ncol=3)

    fig.suptitle('Step 0→1: food_delta vs terrain score\n'
                '(all settlements start with food=0.5, pop=0.5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_step0_terrain.png', dpi=150)
    print("\nSaved food_step0_terrain.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Plot food_delta vs terrain_score for EACH SEED separately
    #         (each seed has constant weather, so should be a PERFECT line)
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("PART 2: Single-seed fits — should be near-perfect")
    print("=" * 80)

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]

        for replay in replays:
            fb, fa = replay["frames"][0], replay["frames"][1]
            grid = fb["grid"]
            seed = replay["seed_index"]

            rows = []
            for pos, sb in {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}.items():
                sa = {(s["x"], s["y"]): s for s in fa["settlements"] if s["alive"]}.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                if grid[y][x] == PORT:
                    continue
                adj = terrain_adj_full(grid, x, y)
                rows.append((
                    adj.get(PLAINS, 0), adj.get(FOREST, 0),
                    adj.get(MOUNTAIN, 0), adj.get(OCEAN, 0),
                    adj.get(SETTLEMENT, 0) + adj.get(PORT, 0),
                    sa["food"] - sb["food"],
                ))

            if not rows:
                continue
            arr = np.array(rows)
            n_pl, n_fo, n_mt, n_oc, n_st = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            fd = arr[:, 5]

            # Fit: food_delta = a + b*pl + c*fo + d*mt + e*oc + f*st
            X = np.column_stack([np.ones(len(fd)), n_pl, n_fo, n_mt, n_oc, n_st])
            c, r2, mae, res = fit_ols(X, fd)

            terrain_score = n_pl + 1.5 * n_fo
            pred = X @ c

            ax.scatter(terrain_score, fd, s=10, alpha=0.4, label=f's={seed}' if seed < 2 else None)
            if seed == 0:
                # plot fitted line
                ts_sorted = np.sort(np.unique(terrain_score))
                ax.set_title(f'Round {rid[:8]} R²(s0)={r2:.4f}')

            if seed == 0:
                print(f"\nRound {rid[:12]} seed={seed}: {len(fd)} settlements")
                print(f"  int={c[0]:.6f} pl={c[1]:.6f} fo={c[2]:.6f} "
                      f"mt={c[3]:.6f} oc={c[4]:.6f} st={c[5]:.6f}")
                print(f"  R²={r2:.6f} MAE={mae:.6f}")
                print(f"  Top residuals:")
                worst = np.argsort(np.abs(res))[-5:]
                for w in worst:
                    print(f"    pl={int(n_pl[w])} fo={int(n_fo[w])} mt={int(n_mt[w])} "
                          f"oc={int(n_oc[w])} st={int(n_st[w])} "
                          f"fd={fd[w]:.6f} pred={pred[w]:.6f} res={res[w]:.6f}")

        ax.set_xlabel('terrain_score')
        ax.set_ylabel('food_delta')
        ax.legend(fontsize=6)

    fig.suptitle('Step 0→1: Each color is a different seed\n'
                '(weather shifts the curve up/down)', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_step0_per_seed.png', dpi=150)
    print("\nSaved food_step0_per_seed.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: After demeaning (removing weather), plot demeaned food_delta
    #         vs pl, fo — should be PERFECTLY clean lines
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("PART 3: Demeaned step 0→1 — remove weather, fit terrain")
    print("=" * 80)

    for rid in sorted(rounds):
        replays = rounds[rid]
        all_fds, all_pls, all_fos, all_sts, all_mts, all_ocs = [], [], [], [], [], []
        group_ids = []

        for replay in replays:
            fb, fa = replay["frames"][0], replay["frames"][1]
            grid = fb["grid"]
            seed = replay["seed_index"]

            for pos, sb in {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}.items():
                sa = {(s["x"], s["y"]): s for s in fa["settlements"] if s["alive"]}.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                if grid[y][x] == PORT:
                    continue
                adj = terrain_adj_full(grid, x, y)
                all_fds.append(sa["food"] - sb["food"])
                all_pls.append(adj.get(PLAINS, 0))
                all_fos.append(adj.get(FOREST, 0))
                all_mts.append(adj.get(MOUNTAIN, 0))
                all_ocs.append(adj.get(OCEAN, 0))
                all_sts.append(adj.get(SETTLEMENT, 0) + adj.get(PORT, 0))
                group_ids.append(seed)

        fd = np.array(all_fds)
        pl = np.array(all_pls)
        fo = np.array(all_fos)
        mt = np.array(all_mts)
        oc = np.array(all_ocs)
        st = np.array(all_sts)
        groups = group_ids

        # Demean by seed
        group_map = defaultdict(list)
        for i, g in enumerate(groups):
            group_map[g].append(i)
        fd_dm = fd.copy()
        for indices in group_map.values():
            fd_dm[indices] -= fd[indices].mean()

        # Also demean features
        pl_dm = pl.copy().astype(float)
        fo_dm = fo.copy().astype(float)
        st_dm = st.copy().astype(float)
        mt_dm = mt.copy().astype(float)
        oc_dm = oc.copy().astype(float)
        for indices in group_map.values():
            pl_dm[indices] -= pl[indices].mean()
            fo_dm[indices] -= fo[indices].mean()
            st_dm[indices] -= st[indices].mean()
            mt_dm[indices] -= mt[indices].mean()
            oc_dm[indices] -= oc[indices].mean()

        # Fit demeaned: food_delta_dm = a*pl_dm + b*fo_dm + c*st_dm
        X = np.column_stack([pl_dm, fo_dm])
        c2, r2_2, mae2, res2 = fit_ols(X, fd_dm)
        print(f"\nRound {rid[:12]}: {len(fd)} obs")
        print(f"  pl + fo only: pl={c2[0]:.8f}, fo={c2[1]:.8f}, "
              f"R²={r2_2:.6f}, MAE={mae2:.8f}, max|res|={np.max(np.abs(res2)):.8f}")
        print(f"  fo/pl = {c2[1]/c2[0]:.6f}")

        X3 = np.column_stack([pl_dm, fo_dm, st_dm])
        c3, r2_3, mae3, res3 = fit_ols(X3, fd_dm)
        print(f"  + st: pl={c3[0]:.8f}, fo={c3[1]:.8f}, st={c3[2]:.8f}, "
              f"R²={r2_3:.6f}, MAE={mae3:.8f}")

        X4 = np.column_stack([pl_dm, fo_dm, st_dm, mt_dm, oc_dm])
        c4, r2_4, mae4, res4 = fit_ols(X4, fd_dm)
        print(f"  + mt + oc: pl={c4[0]:.8f}, fo={c4[1]:.8f}, st={c4[2]:.8f}, "
              f"mt={c4[3]:.8f}, oc={c4[4]:.8f}")
        print(f"    R²={r2_4:.6f}, MAE={mae4:.8f}")


if __name__ == "__main__":
    main()
