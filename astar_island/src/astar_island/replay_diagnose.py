"""Deep diagnostic of worst step-pair mismatches.

Runs replay-eval on all rounds, finds the worst step pairs by LL,
then loads the actual replay frames to show detailed before/after state.

Usage:
    uv run src/astar_island/replay_diagnose.py [--nsims N] [--top N]
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIMULATOR_MANIFEST = Path(__file__).parent.parent.parent / "simulator" / "Cargo.toml"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROJECT_ROOT = Path(__file__).parent.parent.parent

CLASS_NAMES = ["Empty/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
SHORT_CLASS = ["Emp", "Set", "Por", "Rui", "For", "Mnt"]
CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def _sim_cmd(*args: str) -> list[str]:
    return ["cargo", "run", "--release", "--manifest-path", str(SIMULATOR_MANIFEST), "--", *args]


def find_rounds_with_replays() -> list[str]:
    rounds = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "analysis" / "replay_seed_index=0.json").exists():
            rounds.append(d.name)
    return rounds


def run_replay_eval(round_id: str, n_sims: int) -> list[dict]:
    result = subprocess.run(
        _sim_cmd("replay-eval", round_id, str(n_sims)),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(PROJECT_ROOT),
    )
    pairs = []
    for line in result.stdout.strip().split("\n"):
        if line.startswith("PAIR "):
            p = json.loads(line[5:])
            p["round_id"] = round_id
            pairs.append(p)
    return pairs


def load_replay(round_id: str, seed_index: int) -> dict:
    path = DATA_DIR / round_id / "analysis" / f"replay_seed_index={seed_index}.json"
    return json.loads(path.read_text())


def classify_grid(grid):
    """Convert raw grid codes to class indices."""
    return np.vectorize(lambda c: CODE_TO_CLASS.get(c, 0))(np.array(grid))


def describe_settlement(settlements, x, y):
    """Find settlement at (x, y) and describe it."""
    for s in settlements:
        if s["x"] == x and s["y"] == y and s.get("alive", True):
            return (f"pop={s['population']:.3f} food={s['food']:.3f} "
                    f"wealth={s['wealth']:.3f} def={s['defense']:.3f} "
                    f"port={s['has_port']} owner={s['owner_id']}")
    return None


def analyze_step_pair(round_id: str, seed_index: int, step: int, pair_result: dict):
    """Deep analysis of a single step pair."""
    replay = load_replay(round_id, seed_index)
    frame_before = replay["frames"][step]
    frame_after = replay["frames"][step + 1]

    g_before = classify_grid(frame_before["grid"])
    g_after = classify_grid(frame_after["grid"])

    changed_cells = []
    for y in range(40):
        for x in range(40):
            if g_before[y, x] != g_after[y, x]:
                changed_cells.append((y, x, g_before[y, x], g_after[y, x]))

    # Group changes by transition type
    transition_counts = defaultdict(list)
    for y, x, fc, tc in changed_cells:
        transition_counts[(fc, tc)].append((y, x))

    print(f"\n{'='*80}")
    print(f"Round {round_id[:8]}  seed={seed_index}  step={step}->{step+1}")
    print(f"LL={pair_result['grid_ll']:.2f}  changed={pair_result['n_changed']}  modal_acc={pair_result['modal_acc']:.1f}%")
    print(f"Replay ref: data/{round_id}/analysis/replay_seed_index={seed_index}.json  frames[{step}]->[{step+1}]")
    print(f"{'='*80}")

    # Show settlements before
    alive_before = [s for s in frame_before["settlements"] if s.get("alive", True)]
    alive_after = [s for s in frame_after["settlements"] if s.get("alive", True)]
    print(f"\nSettlements: {len(alive_before)} -> {len(alive_after)} ({len(alive_after)-len(alive_before):+d})")

    # Show transition breakdown
    print(f"\nTerrain changes ({len(changed_cells)} cells):")
    for (fc, tc), cells in sorted(transition_counts.items(), key=lambda x: -len(x[1])):
        print(f"  {CLASS_NAMES[fc]} -> {CLASS_NAMES[tc]}: {len(cells)} cells")

        # Show details for the most interesting transitions
        for y, x in cells[:6]:
            # Describe settlement state before and after at this cell
            s_before = describe_settlement(frame_before["settlements"], x, y)
            s_after = describe_settlement(frame_after["settlements"], x, y)

            detail = f"    ({y:2d},{x:2d})"
            if s_before:
                detail += f"  BEFORE: {s_before}"
            if s_after:
                detail += f"\n           AFTER:  {s_after}"
            elif s_before and tc in (0, 3, 4):
                detail += f"\n           AFTER:  (settlement gone)"

            # Find this cell in the worst_cells of the pair result
            for wc in pair_result.get("worst_cells", []):
                if wc["y"] == y and wc["x"] == x:
                    probs = "  ".join(
                        f"{SHORT_CLASS[j]}={wc['sim_probs'][j]:.3f}"
                        for j in range(6) if wc["sim_probs"][j] > 0.001
                    )
                    detail += f"\n           SIM:    [{probs}]  cell_ll={wc['cell_ll']:.2f}"
                    break

            print(detail)

        if len(cells) > 6:
            print(f"    ... and {len(cells) - 6} more")

    # Show nearby context for the most impactful mismatches
    print(f"\nMost impactful sim mismatches (from worst_cells):")
    for wc in pair_result.get("worst_cells", [])[:5]:
        y, x = wc["y"], wc["x"]
        fc, tc = wc["from_class"], wc["expected_class"]
        probs = "  ".join(
            f"{SHORT_CLASS[j]}={wc['sim_probs'][j]:.3f}"
            for j in range(6) if wc["sim_probs"][j] > 0.001
        )
        print(f"  ({y:2d},{x:2d}) {CLASS_NAMES[fc]}->{CLASS_NAMES[tc]}  cell_ll={wc['cell_ll']:.2f}")
        print(f"    sim predicted: [{probs}]")

        # Show neighbors before the step
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40:
                nc = g_before[ny, nx]
                s = describe_settlement(frame_before["settlements"], nx, ny)
                n_str = f"({ny},{nx})={CLASS_NAMES[nc][:3]}"
                if s:
                    n_str += f" [{s}]"
                neighbors.append(n_str)
        print(f"    neighbors before: {'; '.join(neighbors)}")

        # Settlement state if relevant
        s_before = describe_settlement(frame_before["settlements"], x, y)
        s_after = describe_settlement(frame_after["settlements"], x, y)
        if s_before:
            print(f"    settlement before: {s_before}")
        if s_after:
            print(f"    settlement after:  {s_after}")
        elif s_before:
            print(f"    settlement after:  DEAD")


def main():
    args = sys.argv[1:]
    n_sims = 1000
    top_n = 10

    if "--nsims" in args:
        n_sims = int(args[args.index("--nsims") + 1])
    if "--top" in args:
        top_n = int(args[args.index("--top") + 1])

    rounds = find_rounds_with_replays()
    print(f"Running replay-eval on {len(rounds)} rounds with {n_sims} sims/pair...")

    all_pairs = []
    for round_id in rounds:
        pairs = run_replay_eval(round_id, n_sims)
        all_pairs.extend(pairs)
        ll = sum(p["grid_ll"] for p in pairs)
        print(f"  {round_id[:8]}: {len(pairs)} pairs, LL={ll:.2f}")

    total_ll = sum(p["grid_ll"] for p in all_pairs)
    print(f"\nTotal: {len(all_pairs)} pairs, LL={total_ll:.2f}, avg={total_ll/len(all_pairs):.4f}/pair")

    # Find worst step pairs
    all_pairs.sort(key=lambda p: p["grid_ll"])
    print(f"\n{'#'*80}")
    print(f"TOP {top_n} WORST STEP PAIRS")
    print(f"{'#'*80}")

    for i in range(min(top_n, len(all_pairs))):
        p = all_pairs[i]
        analyze_step_pair(p["round_id"], p["seed_index"], p["step"], p)

    # Aggregate transition mismatches
    agg = defaultdict(lambda: [0, 0.0])
    for p in all_pairs:
        for t in p["top_transitions"]:
            key = (t["from_class"], t["to_class"])
            agg[key][0] += t["real_count"]
            agg[key][1] += t["sim_avg"]

    print(f"\n{'#'*80}")
    print("GLOBAL TRANSITION MISMATCHES")
    print(f"{'#'*80}")
    ranked = sorted(agg.items(), key=lambda x: abs(x[1][0] - x[1][1]), reverse=True)
    for (fc, tc), (real, sim) in ranked[:15]:
        delta = real - sim
        direction = "UNDER" if delta > 0 else "OVER"
        print(f"  {CLASS_NAMES[fc]:15s} -> {CLASS_NAMES[tc]:15s}  real={real:7d}  sim={sim:9.1f}  delta={delta:+9.1f}  ({direction})")


if __name__ == "__main__":
    main()
