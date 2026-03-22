import json
from pathlib import Path
from typing import Any
from collections import defaultdict


DATA_DIR = Path("data/parsed_logs_v2")


def load_all_logs() -> dict[str, list[dict[str, Any]]]:
    """Returns {request_id: [log_entries]} for every file."""
    logs = {}
    for f in DATA_DIR.glob("*.json"):
        logs[f.stem] = json.loads(f.read_text())
    return logs


def extract_endpoint_data(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """From a single request's log entry, extract {endpoint: response_data}"""
    result = {}
    for e in entries:
        ext = e.get("extra", {})
        if "data" in ext and "endpoint" in ext:
            result[ext["endpoint"]] = ext["data"]
    return result


def extract_prompt(entries: list[dict[str, Any]]) -> str | None:
    for e in entries:
        ext = e.get("extra", {})
        if "prompt" in ext and e.get("function") == "serve":
            return ext["prompt"]
    return None


def extract_errors(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Returns {endpoint: error_message} for failed API calls."""
    errors = {}
    for e in entries:
        if e.get("level") != "ERROR":
            continue
        ext = e.get("extra", {})
        if "endpoint" in ext and "error" in ext:
            errors[ext["endpoint"]] = ext["error"]
    return errors


def _hashable(val):
    """Convert a value to something hashable for set membership."""
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True)
    return val

def analyze_field_variance(all_responses: dict[str, dict[str, dict[str, Any]]]) -> None:
    """For each endpoint, classify fields as static or varying across requests."""
    # Collect all values seen for each endpoint + field
    endpoint_field_values: dict[str, dict[str, set[Any]]] = defaultdict(lambda: defaultdict(set))

    for _, endpoints in all_responses.items():
        for ep, data in endpoints.items():
            values = data.get("values", [])
            for record in values:
                for key, val in record.items():
                    # Make hashable
                    endpoint_field_values[ep][key].add(_hashable(val))
    
    # Classify
    for ep, fields in sorted(endpoint_field_values.items()):
        print(f"\n=== {ep} ===")
        static = {k: v for k, v in fields.items() if len(v) == 1}
        varying = {k: v for k, v in fields.items() if len(v) > 1}
        print(f"Static fields ({len(static)}):  {list(static.keys())}")
        print(f"Varying fields ({len(varying)}): {list(varying.keys())}")
        for k, vals in varying.items():
            sample = list(vals)[:5]
            print(f"{k}: {len(vals)} unique values, e.g. {sample}")


def show_endpoint_records(all_responses: dict[str, dict[str, dict[str, Any]]], endpoint: str) -> None:
    """Print all unique records seen for an endpoint across all requests."""
    seen_ids = set()
    for _, endpoints in all_responses.items():
        data = endpoints.get(endpoint, {})
        for record in data.get("values", []):
            rec_id = record.get("id")
            if rec_id not in seen_ids:
                seen_ids.add(rec_id)
                print(json.dumps(record, indent=2, ensure_ascii=False))
                print()


if __name__ == "__main__":
    logs = load_all_logs()
    all_responses = {rid: extract_endpoint_data(entries) for rid, entries in logs.items()}

    # see which endpoints are static vs varying across all requests
    analyze_field_variance(all_responses)

    # Drill into a specific endpoint
    show_endpoint_records(all_responses, "/department")
