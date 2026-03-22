from __future__ import annotations

import json
import subprocess
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
VALID_LOG_VERSIONS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def get_logs(use_cached: bool = True) -> list[dict[str, Any]]:
    output_path = DATA_DIR / "logs.json"

    if not use_cached:
        with output_path.open("w") as output_file:
            subprocess.run(
                [
                    "gcloud",
                    "logging",
                    "read",
                    'resource.type="cloud_run_revision" AND resource.labels.service_name="tripletex-agent"',
                    "--project=ai-nm26osl-1729",
                    "--limit=100000",
                    "--format=json",
                ],
                check=True,
                stdout=output_file,
                text=True,
            )

    return json.loads(output_path.read_text())


def extract_log_payload(entry: dict[str, Any]) -> dict[str, Any] | None:
    if entry.get("log_version") in VALID_LOG_VERSIONS:
        return entry

    json_payload = entry.get("jsonPayload")
    if (
        isinstance(json_payload, dict)
        and json_payload.get("log_version") in VALID_LOG_VERSIONS
    ):
        return json_payload

    text_payload = entry.get("textPayload")
    if isinstance(text_payload, str):
        try:
            parsed = json.loads(text_payload)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and parsed.get("log_version") in VALID_LOG_VERSIONS:
            return parsed

    return None


def group_logs_by_request_id(
    log_entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for entry in log_entries:
        payload = extract_log_payload(entry)
        if payload is None:
            continue

        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue

        grouped_logs[request_id].append(payload)

    return dict(grouped_logs)


def write_grouped_logs(
    grouped_logs: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for request_id, logs in grouped_logs.items():
        output_path = output_dir / f"{request_id}.json"
        if request_id in ["-", "a852d92b-d52d-481f-8b50-d3ec753e5bf8"]:
            print(f"WARNING: skipping {request_id=}")
            continue
        output_path.write_text(json.dumps(logs, indent=2))


def main() -> None:
    output_dir = DATA_DIR / "parsed_logs_v2"
    log_entries = get_logs(use_cached=False)
    grouped_logs = group_logs_by_request_id(log_entries)
    write_grouped_logs(grouped_logs, output_dir)

    print(f"Wrote {len(grouped_logs)} request log files to {output_dir}")

    prompts = [
        literal_eval(
            json.loads(Path(file).read_text())[-1]["extra"]["body"]["preview"]
        )["prompt"]
        for file in sorted(list((Path("data") / "parsed_logs_v2").glob("*")))
    ]
    print("\n".join(prompts))


if __name__ == "__main__":
    main()
