from pathlib import Path

import numpy as np
import requests

from astar_island.data import QueryData
from astar_island.data.client import get_active_round_id

n_seeds = 5
height = width = 40

assert Path("pyproject.toml").read_text().splitlines()[1] == 'name = "astar-island"'

(DATA_DIR := Path("data")).mkdir(exist_ok=True)
BASE_API_URL = "https://api.ainm.no"
ROUND_TIME_REMAINING_WARNING_SECS = 120

auth_cookies = {"access_token": Path(".token").read_text().strip()}
# round_id = get_active_round_id()
query_data = QueryData.build()
# +

for seed_idx in range(n_seeds):
    prediction = np.full((height, width, 6), 1 / 6)  # uniform baseline
    for r, row in enumerate(query_data.details.initial_states[seed_idx].grid):
        for c, cell in enumerate(row):
            if cell == 10:
                prediction[r, c] = np.array([1, 0, 0, 0, 0, 0])
            elif cell == 5:
                prediction[r, c] = np.array([0, 0, 0, 0, 0, 1])
    resp = requests.post(
        f"{BASE_API_URL}/astar-island/submit",
        json={
            "round_id": query_data.round_id,
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        },
        cookies=auth_cookies,
    )
    print(f"Seed {seed_idx}: {resp.status_code}")
