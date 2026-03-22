import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

assert Path("pyproject.toml").read_text().splitlines()[1] == 'name = "astar-island"'

(DATA_DIR := Path("data")).mkdir(exist_ok=True)
BASE_API_URL = "https://api.ainm.no"
ROUND_TIME_REMAINING_WARNING_SECS = 120

auth_cookies = {"access_token": Path(".token").read_text().strip()}

_real_requests_enabled: bool | None = None


def ensure_real_requests_enabled() -> bool:
    global _real_requests_enabled
    if _real_requests_enabled is None:
        _real_requests_enabled = input("do you want to make actual requests? [y/N] ") == "y"
    return _real_requests_enabled


def round_data_path(round_id: str) -> Path:
    (path := (DATA_DIR / round_id)).mkdir(exist_ok=True)
    return path


def get_active_round_id(debug=False, override: None | str = None) -> str:
    if override:
        print("OVERRINDING THE ROUND ID!!!", override)
        return override
    rounds = requests.get(f"{BASE_API_URL}/astar-island/rounds", cookies={}).json()
    (active,) = [r for r in rounds if r["status"] == "active"]
    assert active["map_width"] == active["map_height"] and active["map_width"] == 40
    assert active["prediction_window_minutes"] == 165
    # assert active["round_weight"] == 1.05
    closes_at = datetime.fromisoformat(active["closes_at"]).astimezone(
        ZoneInfo("America/Los_Angeles")
    )
    print(closes_at)
    if (
        closes_at - datetime.now(ZoneInfo("America/Los_Angeles"))
    ).total_seconds() < ROUND_TIME_REMAINING_WARNING_SECS:
        print(
            f"\x1b[33mWARNING\x1b[0m: ROUND CLOSES IN LESS THAN {ROUND_TIME_REMAINING_WARNING_SECS}s"
        )
    if debug:
        __import__("rich").print(active)
    return active["id"]


def get_round_details(round_id: str) -> dict:
    detail_path = round_data_path(round_id) / "details.json"
    if detail_path.exists():
        print("loading cached details")
        with detail_path.open("r") as f:
            return json.load(f)
        return json.loads(detail_path.read_text())
    print("redownloading details")
    details = requests.get(
        f"{BASE_API_URL}/astar-island/rounds/{round_id}", cookies=auth_cookies
    ).json()
    with detail_path.open("w") as f:
        json.dump(details, f)

    return details


def get_simulation_result(
    round_id: str, *, map_idx: int, r: int, c: int, run_seed_idx: int
) -> dict:
    assert 0 <= map_idx < 5
    assert 0 <= r < 40
    assert 0 <= c < 40
    # w = {
    #     0: 15,
    #     15: 15,
    #     30: 10,
    # }[c]
    # h = {
    #     0: 15,
    #     15: 15,
    #     30: 10,
    # }[r]
    w = h = 15

    (query_dir := round_data_path(round_id) / "query").mkdir(exist_ok=True)
    query_path = query_dir / f"{map_idx=}_{run_seed_idx=}_{r=}_{c=}_{w=}_{h=}.json"
    if query_path.exists():
        # print(f"found cache for {round_id=} {map_idx=} {r=} {c=}")
        with query_path.open("r") as f:
            return json.load(f)
        assert False

    assert ensure_real_requests_enabled()

    json_data = {
        "round_id": round_id,
        "seed_index": map_idx,
        "viewport_x": c,
        "viewport_y": r,
        "viewport_w": w,
        "viewport_h": h,
    }

    with open("spam.log", "a") as f:
        f.writelines([f"making request with data: {json_data}"])

    time.sleep(1 / 3)
    result = requests.post(
        f"{BASE_API_URL}/astar-island/simulate",
        json=json_data,
        cookies=auth_cookies,
    ).json()

    with query_path.open("w") as f:
        json.dump(result, f)

    with open("spam.log", "a") as f:
        f.writelines([f"got response {result}"])

    return result
