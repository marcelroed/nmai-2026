"""End-to-end pipeline: fetch round, infer params, predict, submit."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

from astar_island.data.client import (
    BASE_API_URL,
    auth_cookies,
    ensure_real_requests_enabled,
    get_active_round_id,
    get_round_details,
    get_simulation_result,
)

SIMULATOR_MANIFEST = Path(__file__).parent.parent.parent / "simulator" / "Cargo.toml"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

def score_prediction(prediction: list, ground_truth: list) -> float:
    """Entropy-weighted KL divergence KL(ground_truth || prediction). Mirrors Rust scoring."""
    total_weighted_kl = 0.0
    total_weight = 0.0
    for gt_row, pred_row in zip(ground_truth, prediction):
        for gt_cell, pred_cell in zip(gt_row, pred_row):
            entropy = sum(-p * math.log(p) for p in gt_cell if p > 0)
            if entropy < 1e-6:
                continue
            kl = sum(
                p * math.log(p / max(q, 1e-10))
                for p, q in zip(gt_cell, pred_cell)
                if p > 0
            )
            total_weighted_kl += entropy * kl
            total_weight += entropy
    return total_weighted_kl / total_weight if total_weight > 0 else 0.0


def competition_score(weighted_kl: float) -> float:
    """Convert weighted KL to 0-100 competition score."""
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * weighted_kl)))


CLASS_NAMES = ["Empty/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def per_class_weighted_kl(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Per-class contribution to entropy-weighted KL. Sum equals total weighted KL."""
    kl_per_class = np.zeros(6)
    total_weight = 0.0
    for gt_row, pred_row in zip(ground_truth, prediction):
        for gt_cell, pred_cell in zip(gt_row, pred_row):
            entropy = sum(-p * math.log(p) for p in gt_cell if p > 0)
            if entropy < 1e-6:
                continue
            total_weight += entropy
            for k in range(6):
                if gt_cell[k] > 0:
                    kl_per_class[k] += entropy * gt_cell[k] * math.log(
                        gt_cell[k] / max(pred_cell[k], 1e-10)
                    )
    if total_weight > 0:
        kl_per_class /= total_weight
    return kl_per_class


def initial_grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    """Convert integer tile-code grid to (H, W, 6) one-hot probability tensor."""
    h, w = grid.shape
    onehot = np.zeros((h, w, 6), dtype=np.float64)
    code_to_class = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    for code, cls in code_to_class.items():
        onehot[grid == code, cls] = 1.0
    return onehot


def per_cell_class_error(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Per-cell, per-class contribution to entropy-weighted KL. Shape (H, W, 6).

    Each value is entropy(gt_cell) * gt[k] * log(gt[k] / pred[k]) / total_weight,
    so summing over all cells and classes gives the total weighted KL.
    """
    gt = ground_truth.astype(np.float64) # (H, W, 6)
    pred = prediction.astype(np.float64) # (H, W, 6)
    
    gt_clamped = np.maximum(gt, 1e-10)
    entropies = -np.sum(np.where(gt > 0, gt * np.log(gt_clamped), 0.0), axis=-1) # (H, W)

    total_weight = np.sum(entropies)
    if total_weight == 0:
        return np.zeros_like(gt)

    pred_clamped = np.maximum(pred, 1e-5)
    error = np.where(
        (gt > 0),
        entropies[..., np.newaxis] * gt * np.log(gt_clamped / pred_clamped) / total_weight,
        0.0,
    )
    return error


def plot_validation(
    seed_idx: int,
    raw: np.ndarray,
    postprocessed: np.ndarray,
    naive: np.ndarray,
    ground_truth: np.ndarray,
    initial_grid: np.ndarray,
    round_id: str,
    raw_score: float,
    pp_score: float,
    naive_score: float,
):
    """Plot initial state, raw, post-processed, ground truth, and error heatmaps."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    raw_kl = per_class_weighted_kl(raw, ground_truth)
    pp_kl = per_class_weighted_kl(postprocessed, ground_truth)
    naive_kl = per_class_weighted_kl(naive, ground_truth)

    initial_onehot = initial_grid_to_onehot(initial_grid)
    pp_error = per_cell_class_error(postprocessed, ground_truth)

    gt_f = ground_truth.astype(np.float64)
    gt_f_clamped = np.maximum(gt_f, 1e-10)
    cell_entropy = -np.sum(np.where(gt_f > 0, gt_f * np.log(gt_f_clamped), 0.0), axis=-1)

    fig = plt.figure(figsize=(24, 22))
    gs = GridSpec(7, 6, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 0.5])

    prob_rows = [
        ("Initial state", initial_onehot),
        ("Raw prediction", raw),
        ("Post-processed", postprocessed),
        ("Ground truth", ground_truth),
    ]
    for row_idx, (label, data) in enumerate(prob_rows):
        for col_idx in range(6):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            channel = data[:, :, col_idx]
            im_prob = ax.imshow(channel, vmin=0, vmax=1, cmap="viridis")
            total_w = float(channel.sum())
            if row_idx == 0:
                ax.set_title(CLASS_NAMES[col_idx], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5, -0.05, f"w={total_w:.1f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
            )

    # Per-cell error row (shared colormap and colorbar)
    error_vmax = max(pp_error[:, :, c].max() for c in range(6))
    error_vmin = min(pp_error[:, :, c].min() for c in range(6))
    error_axes = []
    for col_idx in range(6):
        ax = fig.add_subplot(gs[4, col_idx])
        error_axes.append(ax)
        channel = pp_error[:, :, col_idx]
        im_err = ax.imshow(channel, vmin=error_vmin, vmax=error_vmax, cmap="Reds")
        total_err = float(channel.sum())
        if col_idx == 0:
            ax.set_ylabel("PP cell error", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5, -0.05, f"KL={total_err:.4f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
        )
    # Entropy row (same heatmap repeated 6 times)
    entropy_axes = []
    entropy_vmax = cell_entropy.max()
    for col_idx in range(6):
        ax = fig.add_subplot(gs[5, col_idx])
        entropy_axes.append(ax)
        im_ent = ax.imshow(cell_entropy, vmin=0, vmax=entropy_vmax, cmap="Reds")
        if col_idx == 0:
            ax.set_ylabel("GT entropy", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    # Per-class KL bar chart
    ax_bar = fig.add_subplot(gs[6, :])
    x = np.arange(6)
    width = 0.25
    bars_raw = ax_bar.bar(x - width, raw_kl, width, label=f"Raw (total={raw_kl.sum():.4f})")
    bars_naive = ax_bar.bar(x, naive_kl, width, label=f"Naive (total={naive_kl.sum():.4f})")
    bars_pp = ax_bar.bar(x + width, pp_kl, width, label=f"Post-processed (total={pp_kl.sum():.4f})")
    ax_bar.bar_label(bars_raw, fmt="%.4f", fontsize=7, padding=2)
    ax_bar.bar_label(bars_naive, fmt="%.4f", fontsize=7, padding=2)
    ax_bar.bar_label(bars_pp, fmt="%.4f", fontsize=7, padding=2)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CLASS_NAMES)
    ax_bar.set_ylabel("Weighted KL")
    ax_bar.legend()

    fig.suptitle(
        f"Seed {seed_idx}  |  raw={raw_score:.2f}  naive={naive_score:.2f}  pp={pp_score:.2f}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    # Probability colorbar (rows 0-3)
    cbar_ax_prob = fig.add_axes([0.89, 0.45, 0.012, 0.48])
    fig.colorbar(im_prob, cax=cbar_ax_prob, label="Probability")
    # Error colorbar (row 4)
    cbar_ax_err = fig.add_axes([0.89, 0.28, 0.012, 0.13])
    fig.colorbar(im_err, cax=cbar_ax_err, label="Cell KL error")
    # Entropy colorbar (row 5)
    cbar_ax_ent = fig.add_axes([0.89, 0.12, 0.012, 0.13])
    fig.colorbar(im_ent, cax=cbar_ax_ent, label="Entropy (nats)")

    out_dir = DATA_DIR / round_id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"validation_seed_{seed_idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def naive_postprocess(prediction: np.ndarray, initial_grid: np.ndarray, ruin_fraction: float) -> np.ndarray:
    """Naive postprocessing: set ruins = ruin_fraction * settlements, then project to feasible."""
    from astar_island.postprocess import postprocess

    return postprocess(prediction, initial_grid, ruin_fraction)


def validate_predictions(
    round_id: str,
    raw_predictions: dict[int, list],
    predictions: dict[int, list],
    initial_states: list[dict],
):
    """Validate predictions against ground truth if available, with plots."""
    from astar_island.postprocess import compute_ruin_fraction

    analysis_dir = DATA_DIR / round_id / "analysis"
    if not analysis_dir.exists():
        print("No ground truth data available for this round, skipping validation.")
        return

    ruin_fraction = compute_ruin_fraction(round_id)

    scores = []
    for seed_idx in sorted(predictions.keys()):
        gt_path = analysis_dir / f"ground_truth_seed_index={seed_idx}.json"
        if not gt_path.exists():
            continue
        gt_data = json.loads(gt_path.read_text())
        ground_truth = gt_data["ground_truth"]

        raw_arr = np.array(raw_predictions[seed_idx])
        grid = np.array(initial_states[seed_idx]["grid"])
        naive_arr = naive_postprocess(raw_arr, grid, ruin_fraction)

        raw_kl = score_prediction(raw_predictions[seed_idx], ground_truth)
        raw_cs = competition_score(raw_kl)
        naive_kl = score_prediction(naive_arr.tolist(), ground_truth)
        naive_cs = competition_score(naive_kl)
        pp_kl = score_prediction(predictions[seed_idx], ground_truth)
        pp_cs = competition_score(pp_kl)
        scores.append(pp_cs)
        print(f"  seed {seed_idx}: raw={raw_cs:.2f}  naive={naive_cs:.2f}  pp={pp_cs:.2f}")

        plot_validation(
            seed_idx,
            raw=raw_arr,
            postprocessed=np.array(predictions[seed_idx]),
            naive=naive_arr,
            ground_truth=np.array(ground_truth),
            initial_grid=grid,
            round_id=round_id,
            raw_score=raw_cs,
            pp_score=pp_cs,
            naive_score=naive_cs,
        )

    if scores:
        avg = sum(scores) / len(scores)
        print(f"  average score: {avg:.2f}/100 ({len(scores)} seeds)")
    else:
        print("No ground truth files found for any seed.")


def submit_prediction(round_id: str, seed_index: int, prediction: list):
    """Submit a prediction tensor for one seed."""
    resp = requests.post(
        f"{BASE_API_URL}/astar-island/submit",
        json={"round_id": round_id, "seed_index": seed_index, "prediction": prediction},
        cookies=auth_cookies,
    )
    resp.raise_for_status()
    return resp.json()


def run_inference(round_id: str, budget: int = 1) -> str | None:
    """Run parameter inference on cached query observations. Returns params JSON string."""
    print(f"Running parameter inference for {round_id} (budget x{budget})...")
    cmd = ["cargo", "run", "--release", "--manifest-path", str(SIMULATOR_MANIFEST), "--", "infer", round_id]
    if budget > 1:
        cmd.append(str(budget))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    if result.returncode != 0:
        print("Inference failed (see stderr above)")
        return None
    params_json = result.stdout.strip()
    print(f"Inferred params: {params_json}")
    return params_json


def run_queries(round_id: str, details: dict) -> int:
    """Run planned viewport queries, caching results locally. Returns number executed."""
    from astar_island.query_planner import plan_queries
    import requests as req
    from astar_island.data.client import BASE_API_URL, auth_cookies

    # Check budget
    budget = req.get(f"{BASE_API_URL}/astar-island/budget", cookies=auth_cookies).json()
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining <= 0:
        print(f"Query budget exhausted ({budget['queries_used']}/{budget['queries_max']})")
        return 0

    queries = plan_queries(details["initial_states"], budget=remaining)
    print(f"Running {len(queries)} queries (budget: {remaining} remaining)...")

    executed = 0
    for i, q in enumerate(queries):
        result = get_simulation_result(
            round_id,
            map_idx=q["seed_index"],
            r=q["viewport_y"],
            c=q["viewport_x"],
            run_seed_idx=i,
        )
        executed += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(queries)} queries done")

    print(f"Executed {executed} queries")
    return executed


def run_pipeline(round_id: str | None = None, n_sims: int = 2000, use_inference: bool = False, validate: bool = False, submit: bool = False, infer_budget: int = 1):
    """Full pipeline: fetch, query, optionally infer params, predict via Rust, submit."""
    if round_id is None:
        round_id = get_active_round_id()

    details = get_round_details(round_id)
    print(f"Round {details['round_number']} ({round_id})")

    # Step 0: Run queries if budget available and real requests are enabled
    if ensure_real_requests_enabled():
        run_queries(round_id, details)

    # Step 1: Optionally run inference
    params_json = None
    if use_inference:
        params_json = run_inference(round_id, budget=infer_budget)
        if params_json is None:
            print("Inference failed, falling back to default params")

    # Step 2: Run Monte Carlo via Rust binary
    use_ensemble = use_inference and params_json is None  # ensemble when inference had queries
    if use_ensemble:
        print(f"Running ensemble Monte Carlo ({n_sims} sims per seed, top-10)...")
        cmd = ["cargo", "run", "--release", "--manifest-path", str(SIMULATOR_MANIFEST), "--", "ensemble", round_id, str(n_sims), "10"]
    else:
        print(f"Running Monte Carlo ({n_sims} sims per seed)...")
        cmd = ["cargo", "run", "--release", "--manifest-path", str(SIMULATOR_MANIFEST), "--", "montecarlo", round_id, str(n_sims)]
        if params_json:
            cmd.append(params_json)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    if result.returncode != 0:
        print("Simulator failed (see stderr above)")
        return

    # Step 3: Parse output (SEED_N: {json})
    predictions: dict[int, list] = {}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("SEED_"):
            parts = line.split(":", 1)
            seed_idx = int(parts[0].split("_")[1])
            predictions[seed_idx] = json.loads(parts[1])

    # Step 4: Post-process predictions
    from astar_island.postprocess import postprocess_predictions

    raw_predictions = predictions
    print("Post-processing predictions...")
    predictions = postprocess_predictions(predictions, details["initial_states"], round_id)

    # Step 5: Validate against ground truth if requested
    if validate:
        print("Validating predictions against ground truth...")
        validate_predictions(round_id, raw_predictions, predictions, details["initial_states"])

    # Step 6: Submit predictions
    if submit:
        for seed_idx in range(details["seeds_count"]):
            if seed_idx in predictions:
                print(f"Submitting seed {seed_idx}...")
                resp = submit_prediction(round_id, seed_idx, predictions[seed_idx])
                print(f"  Response: {resp}")


if __name__ == "__main__":
    round_id = sys.argv[1] if len(sys.argv) > 1 else None
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    use_inference = "--infer" in sys.argv
    validate = "--validate" in sys.argv
    submit = "--submit" in sys.argv
    infer_budget = 1
    if "--infer-budget" in sys.argv:
        idx = sys.argv.index("--infer-budget")
        infer_budget = int(sys.argv[idx + 1])
    run_pipeline(round_id, n_sims, use_inference, validate, submit, infer_budget=infer_budget)
