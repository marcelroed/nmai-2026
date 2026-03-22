import json
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import Literal

import polars as pl
import requests
from pydantic import BaseModel, Field, model_validator
from pydantic.dataclasses import dataclass
from rich.pretty import install

from tripletex.config import AUTH, BASE_URL

# +


class DepartmentValues(BaseModel):
    id: int
    name: str
    url: str
    displayName: str
    departmentNumber: str
    department_manager: None = Field(alias="departmentManager")


class DepartmentResult(BaseModel):
    count: int
    fullResultSize: int
    values: list[DepartmentValues]
    endpoint: Literal["/department"]
    from_: int = Field(alias="from")

    @model_validator(mode="before")
    @classmethod
    def parse_raw_data(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(
            data.get("data") if isinstance(data.get("data"), dict) else data
        )
        normalized["endpoint"] = data["endpoint"]
        return normalized


@dataclass
class ParsedData:
    prompt: str
    task: str
    files: list[dict]
    request_id: str
    department: DepartmentResult
    log_version: int


def get_log_version(x):
    (log_version,) = set([y["log_version"] for y in x])
    return log_version


def get_all_data() -> tuple[dict[str, ParsedData], dict]:
    prompt_to_task_df = pl.read_csv("data/tasks/task_1_2_3.csv")
    data = {}
    for log_path in (Path("data") / "parsed_logs_v2").glob("*"):
        data[log_path.name.replace(".json", "")] = json.loads(log_path.read_text())
    parsed_data: dict[str, ParsedData] = {}

    def get_department_data(request_id) -> DepartmentResult | None:

        for i, log_entry in enumerate(data[request_id]):
            try:
                return DepartmentResult.model_validate(log_entry["extra"])
            except Exception as _:
                pass
        return None

    for request_id in data:
        body = literal_eval(data[request_id][-1]["extra"]["body"]["preview"])
        prompt = body["prompt"]
        (task,) = prompt_to_task_df.filter(pl.col("prompt") == prompt)["task"].unique()
        if department_result := get_department_data(request_id):
            parsed_data[request_id] = ParsedData(
                prompt=prompt,
                task=task,
                files=body["files"],
                request_id=request_id,
                department=department_result,
                log_version=get_log_version(data[request_id]),
            )
        else:
            print(f"skipping {prompt} {request_id}")
    return parsed_data, data


def get_tasks() -> tuple[dict[str, list[ParsedData]], dict]:
    tasks: dict[str, list[ParsedData]] = defaultdict(list)
    parsed_data, raw_data = get_all_data()

    for request_id in parsed_data:
        _data = parsed_data[request_id]
        tasks[_data.task.split(":")[0]].append(_data)
    return tasks, raw_data


def get_task(task: str, only_log_version_2: bool) -> tuple[list[ParsedData], dict]:
    all_tasks, raw_data = get_tasks()
    tasks = all_tasks[task]
    if only_log_version_2:
        tasks = [task for task in tasks if task.log_version == 2]
    assert tasks
    return tasks, {
        request_id: v
        for request_id, v in raw_data.items()
        if request_id in [task.request_id for task in tasks]
    }


#
# # +
# sorted_tasks = sorted(tasks, key=lambda x: int(x.split()[1]))
#
# print()
# print("n : task")
# for task in sorted_tasks:
#     print(f"{task.split(':')[0]:<7} has {len(tasks[task])} tasks")
#     try:
#         print([task for task in tasks[task] if task.log_version == 2][0].prompt)
#     except Exception as _:
#         print(tasks[task][0].prompt)
#     print()
#
# # +
#
# request_id = [task for task in tasks["Task 9"] if task.log_version == 2][0].request_id
#
# parsed_data[request_id].prompt
# data[request_id]
# [(i, x["message"]) for i, x in enumerate(data[request_id][::-1])]
#
# # +
#
# [(i, x["message"]) for i, x in enumerate(data["900745a0-bc4f-483a-8683-9101e8cc1ad2"])]
# [x for x in parsed_data.values() if x.log_version == 2][0].request_id
#
# # +
#
#
# # +
#
# set([x["log_version"] for x in data["900745a0-bc4f-483a-8683-9101e8cc1ad2"]])
#
#
# [get_log_version(x) for x in data.values()]
#
#
# # +
#
# (invoice,) = [
#     x
#     for x in data["900745a0-bc4f-483a-8683-9101e8cc1ad2"]
#     if "get tripletex data for /invoice" in x["message"]
# ]
# invoice
# # +
#
# [
#     x["extra"]["endpoint"]
#     for x in data["900745a0-bc4f-483a-8683-9101e8cc1ad2"]
#     if "failed to get tripletex data for" in x["message"]
# ]
#
#
# # +
#
# requests.get(
#     f"{BASE_URL}/invoice",
#     auth=AUTH,
#     params={"invoiceDateFrom": "1970-01-01", "invoiceDateTo": "2027-12-31"},
# ).json()
#
# # +
#
#
# len(
#     str(
#         requests.get(
#             f"{BASE_URL}/ledger/account",
#             auth=AUTH,
#         ).json()
#     )
# )
#
# # +
#
# endpoint = "/ledger/voucher"
# res = requests.get(
#     f"{BASE_URL}{endpoint}",
#     auth=AUTH,
#     params={"dateFrom": "1970-01-01", "dateTo": "2027-12-31"},
# ).json()
# res
#
# # +
#
# for endpoint in [
#     "/ledger",
#     "/ledger/posting",
#     "/ledger/voucher",
# ]:
#     try:
#         print(endpoint)
#         res = requests.get(
#             f"{BASE_URL}{endpoint}",
#             auth=AUTH,
#         ).json()
#         if status := res.get("status"):
#             print(status)
#     except Exception as _:
#         print("fail for", endpoint)
#     print()
