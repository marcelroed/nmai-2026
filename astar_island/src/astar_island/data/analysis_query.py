import json
import time
from datetime import datetime
from genericpath import exists
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import rich

from astar_island.data.client import (
    BASE_API_URL,
    auth_cookies,
    ensure_real_requests_enabled,
    get_round_details,
    round_data_path,
)

# from astar_island.simulator.ground_truth import query_data

# assert Path("pyproject.toml").read_text().splitlines()[1] == 'name = "astar-island"'
#
# (DATA_DIR := Path("data")).mkdir(exist_ok=True)
# BASE_API_URL = "https://api.ainm.no"
# ROUND_TIME_REMAINING_WARNING_SECS = 120
#
# auth_cookies = {"access_token": Path(".token").read_text().strip()}
#
# _real_requests_enabled = False

ERROR_DETAIL_KEY = "detail"


def get_analysis_dir(round_id: str) -> Path:
    (analysis_dir := round_data_path(round_id) / "analysis").mkdir(exist_ok=True)
    return analysis_dir


def is_error_payload(data: dict) -> bool:
    return set(data) == {ERROR_DETAIL_KEY}


def get_analysis(
    round_id: str, *, seed_index: int, force_refresh: bool = False
) -> dict:
    analysis_path = get_analysis_dir(round_id) / f"ground_truth_{seed_index=}.json"
    if analysis_path.exists() and not force_refresh:
        return json.loads(analysis_path.read_text())

    data = requests.get(
        f"https://api.ainm.no/astar-island/analysis/{round_id}/{seed_index}",
        cookies=auth_cookies,
    ).json()

    if not is_error_payload(data):
        with analysis_path.open("w") as f:
            json.dump(data, f)

    return data


def get_replay(round_id: str, seed_index: int, force_refresh: bool = False) -> dict:

    analysis_path = get_analysis_dir(round_id) / f"replay_{seed_index=}.json"
    if analysis_path.exists() and not force_refresh:
        return json.loads(analysis_path.read_text())

    json_data = {
        "round_id": round_id,
        "seed_index": seed_index,
    }

    data = requests.post(
        "https://api.ainm.no/astar-island/replay", cookies=auth_cookies, json=json_data
    ).json()

    if not is_error_payload(data):
        with analysis_path.open("w") as f:
            json.dump(data, f)

    return data


def iter_error_analysis_paths() -> list[Path]:
    error_paths: list[Path] = []
    for analysis_path in sorted(Path("data").glob("*/analysis/*.json")):
        data = json.loads(analysis_path.read_text())
        if is_error_payload(data):
            error_paths.append(analysis_path)
    return error_paths


def refetch_error_files(*, sleep_seconds: float = 1.0) -> list[tuple[Path, dict]]:
    error_paths = iter_error_analysis_paths()
    refetched: list[tuple[Path, dict]] = []

    for idx, analysis_path in enumerate(error_paths):
        round_id = analysis_path.parent.parent.name
        seed_index = int(analysis_path.stem.split("=")[-1])
        analysis_path.unlink(missing_ok=True)

        if analysis_path.name.startswith("ground_truth_"):
            data = get_analysis(
                round_id=round_id, seed_index=seed_index, force_refresh=True
            )
        else:
            data = get_replay(
                round_id=round_id, seed_index=seed_index, force_refresh=True
            )

        refetched.append((analysis_path, data))

        if idx + 1 < len(error_paths):
            time.sleep(sleep_seconds)

    return refetched


if __name__ == "__main__":

    def main():
        for round_id in [
            "71451d74-be9f-471f-aacd-a41f3b68a9cd",
            "76909e29-f664-4b2f-b16b-61b7507277e9",
            "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
            "8e839974-b13b-407b-a5e7-fc749d877195",
            "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
            "ae78003a-4efe-425a-881a-d16a39bca0ad",
            "36e581f1-73f8-453f-ab98-cbe3052b701b",
            "c5cdf100-a876-4fb7-b5d8-757162c97989",
            "2a341ace-0f57-4309-9b89-e59fe0f09179",
        ]:
            for seed in range(5):
                try:
                    print(f"{round_id=} {seed=}")
                    print(f"{len(get_round_details(round_id))=}")
                    print(f"{len(get_analysis(round_id=round_id, seed_index=seed))=}")
                    print(
                        f"{len(get_replay(round_id=round_id, seed_index=seed).keys())=}"
                    )
                    print()
                except:
                    print(f"FAILED FOR {round_id=} {seed=}")

    main()
