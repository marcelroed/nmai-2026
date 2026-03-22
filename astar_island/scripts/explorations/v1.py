from pathlib import Path

import requests
import rich

BASE = "https://api.ainm.no"

jwt_token = Path(".token").read_text().strip()
cookies = {"access_token": jwt_token}

# +

rounds = requests.get(f"{BASE}/astar-island/rounds", cookies={}).json()
(active,) = (r for r in rounds if r["status"] == "active")
round_id = active["id"]
rich.print(active)

# +

active["id"]

detail = requests.get(f"{BASE}/astar-island/rounds/{round_id}", cookies=cookies).json()

width = detail["map_width"]  # 40
height = detail["map_height"]  # 40
seeds = detail["seeds_count"]  # 5
print(f"Round: {width}x{height}, {seeds} seeds")

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]  # height x width terrain codes
    settlements = state["settlements"]  # [{x, y, has_port, alive}, ...]
    print(f"Seed {i}: {len(settlements)} settlements")

# +

rich.print(detail)
