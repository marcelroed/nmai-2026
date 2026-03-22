import numpy as np

from astar_island.data.client import get_active_round_id

n_seeds = 5
height = width = 40
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo

import requests

assert Path("pyproject.toml").read_text().splitlines()[1] == 'name = "astar-island"'

(DATA_DIR := Path("data")).mkdir(exist_ok=True)
BASE_API_URL = "https://api.ainm.no"
ROUND_TIME_REMAINING_WARNING_SECS = 120

auth_cookies = {"access_token": Path(".token").read_text().strip()}
round_id = get_active_round_id()

for seed_idx in range(n_seeds):
    prediction = np.full((height, width, 6), 1 / 6)  # uniform baseline
    resp = requests.post(
        f"{BASE_API_URL}/astar-island/submit",
        json={
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        },
        cookies=auth_cookies,
    )
    print(f"Seed {seed_idx}: {resp.status_code}")
