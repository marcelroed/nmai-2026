"""Shared utility: load all replay data into a flat DataFrame of step transitions."""
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TERRAIN_NAMES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}

def load_all_replays():
    """Load all replay JSON files, return list of (round_id, seed_index, frames)."""
    replays = []
    for path in sorted(DATA_DIR.glob("*/analysis/replay_seed_index=*.json")):
        with open(path) as f:
            d = json.load(f)
        replays.append((d["round_id"], d["seed_index"], d["frames"]))
    print(f"Loaded {len(replays)} replays")
    return replays


def count_adjacent_terrain(grid, x, y):
    """Count terrain types in the 8-connected neighborhood of (x,y)."""
    h, w = len(grid), len(grid[0])
    counts = {}
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                t = grid[ny][nx]
                counts[t] = counts.get(t, 0) + 1
    return counts


def build_transition_df(replays):
    """Build a DataFrame with one row per (settlement, step) transition.

    Columns: round_id, seed, step, x, y, has_port, owner_id,
             pop, food, wealth, defense, alive,
             pop_next, food_next, wealth_next, defense_next, alive_next, has_port_next, owner_id_next,
             d_pop, d_food, d_wealth, d_defense,
             n_plains, n_forest, n_mountain, n_ocean, n_settlement, n_port, n_ruin, n_empty
    """
    rows = []
    for round_id, seed, frames in replays:
        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            step = f0["step"]
            grid0 = f0["grid"]

            # Index settlements by (x, y) in both frames
            s0_map = {(s["x"], s["y"]): s for s in f0["settlements"]}
            s1_map = {(s["x"], s["y"]): s for s in f1["settlements"]}

            for (x, y), s in s0_map.items():
                if not s["alive"]:
                    continue
                adj = count_adjacent_terrain(grid0, x, y)

                if (x, y) in s1_map:
                    s1 = s1_map[(x, y)]
                    row = {
                        "round_id": round_id[:8], "seed": seed, "step": step,
                        "x": x, "y": y, "has_port": s["has_port"], "owner_id": s["owner_id"],
                        "pop": s["population"], "food": s["food"],
                        "wealth": s["wealth"], "defense": s["defense"],
                        "alive": s["alive"],
                        "pop_next": s1["population"], "food_next": s1["food"],
                        "wealth_next": s1["wealth"], "defense_next": s1["defense"],
                        "alive_next": s1["alive"], "has_port_next": s1["has_port"],
                        "owner_id_next": s1["owner_id"],
                        "d_pop": s1["population"] - s["population"],
                        "d_food": s1["food"] - s["food"],
                        "d_wealth": s1["wealth"] - s["wealth"],
                        "d_defense": s1["defense"] - s["defense"],
                        "n_plains": adj.get(11, 0), "n_forest": adj.get(4, 0),
                        "n_mountain": adj.get(5, 0), "n_ocean": adj.get(10, 0),
                        "n_settlement": adj.get(1, 0), "n_port": adj.get(2, 0),
                        "n_ruin": adj.get(3, 0), "n_empty": adj.get(0, 0),
                    }
                    rows.append(row)
                else:
                    # Settlement died (not in next frame means it became a ruin)
                    row = {
                        "round_id": round_id[:8], "seed": seed, "step": step,
                        "x": x, "y": y, "has_port": s["has_port"], "owner_id": s["owner_id"],
                        "pop": s["population"], "food": s["food"],
                        "wealth": s["wealth"], "defense": s["defense"],
                        "alive": s["alive"],
                        "pop_next": 0, "food_next": 0,
                        "wealth_next": 0, "defense_next": 0,
                        "alive_next": False, "has_port_next": False,
                        "owner_id_next": s["owner_id"],
                        "d_pop": -s["population"], "d_food": -s["food"],
                        "d_wealth": -s["wealth"], "d_defense": -s["defense"],
                        "n_plains": adj.get(11, 0), "n_forest": adj.get(4, 0),
                        "n_mountain": adj.get(5, 0), "n_ocean": adj.get(10, 0),
                        "n_settlement": adj.get(1, 0), "n_port": adj.get(2, 0),
                        "n_ruin": adj.get(3, 0), "n_empty": adj.get(0, 0),
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Built transition DataFrame: {len(df)} rows")
    return df


def build_spawn_df(replays):
    """Build a DataFrame of new settlement spawn events.

    A spawn is detected when a settlement exists at step t+1 but not at step t.
    Returns info about the child AND potential parent (nearest same-owner settlement).
    """
    rows = []
    for round_id, seed, frames in replays:
        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            step = f0["step"]
            grid0 = f0["grid"]
            grid1 = f1["grid"]

            s0_pos = {(s["x"], s["y"]) for s in f0["settlements"]}
            s0_map = {(s["x"], s["y"]): s for s in f0["settlements"]}
            s1_map = {(s["x"], s["y"]): s for s in f1["settlements"]}

            # New settlements: in frame 1 but not frame 0
            for (x, y), s1 in s1_map.items():
                if (x, y) in s0_pos:
                    continue
                if not s1["alive"]:
                    continue

                # This is a new settlement
                terrain_before = grid0[y][x]  # what terrain was here before
                terrain_after = grid1[y][x]

                # Find potential parent: nearest same-owner alive settlement in frame 0
                parent = None
                min_dist = float("inf")
                for (px, py), ps in s0_map.items():
                    if not ps["alive"]:
                        continue
                    if ps["owner_id"] != s1["owner_id"]:
                        continue
                    dist = max(abs(px - x), abs(py - y))  # Chebyshev
                    if dist < min_dist:
                        min_dist = dist
                        parent = ps

                # Also find parent's state in frame 1
                parent_next = None
                if parent:
                    parent_next = s1_map.get((parent["x"], parent["y"]))

                # Count adjacent terrain of child in frame 0
                adj = count_adjacent_terrain(grid0, x, y)

                row = {
                    "round_id": round_id[:8], "seed": seed, "step": step,
                    "child_x": x, "child_y": y,
                    "child_pop": s1["population"], "child_food": s1["food"],
                    "child_wealth": s1["wealth"], "child_defense": s1["defense"],
                    "child_has_port": s1["has_port"], "child_owner_id": s1["owner_id"],
                    "terrain_before": terrain_before,
                    "terrain_before_name": TERRAIN_NAMES.get(terrain_before, str(terrain_before)),
                    "terrain_after": grid1[y][x],
                    "parent_dist": min_dist if parent else None,
                    "parent_x": parent["x"] if parent else None,
                    "parent_y": parent["y"] if parent else None,
                    "parent_pop": parent["population"] if parent else None,
                    "parent_food": parent["food"] if parent else None,
                    "parent_wealth": parent["wealth"] if parent else None,
                    "parent_defense": parent["defense"] if parent else None,
                    "parent_has_port": parent["has_port"] if parent else None,
                    "parent_pop_next": parent_next["population"] if parent_next else None,
                    "parent_food_next": parent_next["food"] if parent_next else None,
                    "parent_wealth_next": parent_next["wealth"] if parent_next else None,
                    "parent_defense_next": parent_next["defense"] if parent_next else None,
                    "parent_alive_next": parent_next["alive"] if parent_next else None,
                    "n_adj_ocean": adj.get(10, 0),
                    "n_adj_plains": adj.get(11, 0),
                    "n_adj_forest": adj.get(4, 0),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Built spawn DataFrame: {len(df)} rows")
    return df


def build_raid_df(replays):
    """Build a DataFrame of raid events by detecting defense drops.

    A raid is detected when a settlement's defense drops significantly in one step.
    We pair potential attacker-defender relationships.
    """
    rows = []
    for round_id, seed, frames in replays:
        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            step = f0["step"]

            s0_map = {(s["x"], s["y"]): s for s in f0["settlements"]}
            s1_map = {(s["x"], s["y"]): s for s in f1["settlements"]}

            # Find defenders: settlements with defense drop
            for (x, y), s in s0_map.items():
                if not s["alive"]:
                    continue
                if (x, y) not in s1_map:
                    continue
                s1 = s1_map[(x, y)]

                d_def = s1["defense"] - s["defense"]
                if d_def >= 0:
                    continue  # No raid detected

                # Check if ownership changed (successful raid)
                owner_changed = s1["owner_id"] != s["owner_id"]

                # Find potential attacker: nearest different-owner settlement in frame 0
                attacker = None
                min_dist = float("inf")
                for (ax, ay), a in s0_map.items():
                    if not a["alive"]:
                        continue
                    if a["owner_id"] == s["owner_id"]:
                        continue
                    dist = ((ax - x)**2 + (ay - y)**2)**0.5  # Euclidean
                    if dist < min_dist:
                        min_dist = dist
                        attacker = a

                attacker_next = None
                if attacker:
                    attacker_next = s1_map.get((attacker["x"], attacker["y"]))

                row = {
                    "round_id": round_id[:8], "seed": seed, "step": step,
                    # Defender
                    "def_x": x, "def_y": y,
                    "def_pop": s["population"], "def_food": s["food"],
                    "def_wealth": s["wealth"], "def_defense": s["defense"],
                    "def_has_port": s["has_port"], "def_owner_id": s["owner_id"],
                    "def_pop_next": s1["population"], "def_food_next": s1["food"],
                    "def_wealth_next": s1["wealth"], "def_defense_next": s1["defense"],
                    "def_alive_next": s1["alive"], "def_owner_id_next": s1["owner_id"],
                    "d_def_pop": s1["population"] - s["population"],
                    "d_def_food": s1["food"] - s["food"],
                    "d_def_wealth": s1["wealth"] - s["wealth"],
                    "d_def_defense": d_def,
                    "owner_changed": owner_changed,
                    # Attacker (nearest enemy)
                    "atk_dist": min_dist if attacker else None,
                    "atk_pop": attacker["population"] if attacker else None,
                    "atk_food": attacker["food"] if attacker else None,
                    "atk_wealth": attacker["wealth"] if attacker else None,
                    "atk_defense": attacker["defense"] if attacker else None,
                    "atk_has_port": attacker["has_port"] if attacker else None,
                    "atk_owner_id": attacker["owner_id"] if attacker else None,
                    "atk_pop_next": attacker_next["population"] if attacker_next else None,
                    "atk_food_next": attacker_next["food"] if attacker_next else None,
                    "atk_wealth_next": attacker_next["wealth"] if attacker_next else None,
                    "atk_defense_next": attacker_next["defense"] if attacker_next else None,
                    "atk_alive_next": attacker_next["alive"] if attacker_next else None,
                    "d_atk_pop": (attacker_next["population"] - attacker["population"]) if (attacker and attacker_next) else None,
                    "d_atk_food": (attacker_next["food"] - attacker["food"]) if (attacker and attacker_next) else None,
                    "d_atk_wealth": (attacker_next["wealth"] - attacker["wealth"]) if (attacker and attacker_next) else None,
                    "d_atk_defense": (attacker_next["defense"] - attacker["defense"]) if (attacker and attacker_next) else None,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Built raid DataFrame: {len(df)} rows")
    return df


if __name__ == "__main__":
    replays = load_all_replays()
    df = build_transition_df(replays)
    print(df.describe())
    print("\nSample:")
    print(df.head())
