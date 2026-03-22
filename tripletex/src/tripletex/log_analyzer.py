"""Analyze parsed logs from data/parsed_logs_v2/ into structured request summaries."""

import json
import os
import re
import sys
from pathlib import Path


def parse_log_file(path: str) -> dict:
    with open(path) as f:
        entries = json.load(f)

    request_id = Path(path).stem

    # Sort by timestamp (logs may be out of order)
    entries.sort(key=lambda e: e.get("timestamp", ""))

    result = {
        "request_id": request_id,
        "prompt": None,
        "files": [],
        "base_url": None,
        "session_token": None,
        "status_code": None,
        "duration_ms": None,
        "timestamp_start": None,
        "timestamp_end": None,
        "endpoints": [],
    }

    for e in entries:
        msg = e["message"].split("] ", 1)[-1] if "] " in e["message"] else e["message"]
        extra = e.get("extra", {})
        level = e["level"]

        # --- Request lifecycle ---
        if msg == "request.start":
            result["timestamp_start"] = e.get("timestamp")

        elif msg == "request.end":
            result["timestamp_end"] = e.get("timestamp")
            result["status_code"] = extra.get("status_code")
            result["duration_ms"] = extra.get("duration_ms")

        elif msg == "serve.handler.enter":
            result["prompt"] = extra.get("prompt")
            result["base_url"] = extra.get("base_url")
            file_count = extra.get("file_count", 0)

            # Extract files info from payload string
            payload_str = extra.get("payload", "")
            if isinstance(payload_str, str):
                # Extract filenames
                for m in re.finditer(r"filename='([^']+)'", payload_str):
                    result["files"].append(m.group(1))
                # Extract session token
                m = re.search(r"session_token='([^']+)'", payload_str)
                if m:
                    result["session_token"] = m.group(1)

        # --- Endpoint calls: getting (request start) ---
        elif msg.startswith("getting data for "):
            endpoint = msg.replace("getting data for ", "")
            # Will be paired with success/failure below

        elif msg.startswith("w/params data for "):
            endpoint = msg.replace("w/params data for ", "")

        # --- Endpoint calls: success ---
        elif msg.startswith("got tripletex data for "):
            endpoint = msg.replace("got tripletex data for ", "")
            ep_entry = {
                "endpoint": extra.get("endpoint", endpoint),
                "status": "success",
                "params": extra.get("params"),
                "data": extra.get("data"),
            }
            result["endpoints"].append(ep_entry)

        elif msg.startswith("w/params got tripletex data for "):
            endpoint = msg.replace("w/params got tripletex data for ", "")
            ep_entry = {
                "endpoint": extra.get("endpoint", endpoint),
                "status": "success",
                "params": extra.get("params"),
                "data": extra.get("data"),
            }
            result["endpoints"].append(ep_entry)

        # --- Endpoint calls: failure ---
        elif msg.startswith("failed to get tripletex data for "):
            endpoint = msg.replace("failed to get tripletex data for ", "")
            ep_entry = {
                "endpoint": extra.get("endpoint", endpoint),
                "status": "error",
                "error": extra.get("error"),
                "params": extra.get("params"),
            }
            result["endpoints"].append(ep_entry)

        elif msg.startswith("w/params failed to get tripletex data for "):
            endpoint = msg.replace("w/params failed to get tripletex data for ", "")
            ep_entry = {
                "endpoint": extra.get("endpoint", endpoint),
                "status": "error",
                "error": extra.get("error"),
                "params": extra.get("params"),
            }
            result["endpoints"].append(ep_entry)

    return result


def summarize_endpoints(endpoints: list[dict]) -> dict:
    """Create a compact summary of endpoint results."""
    success = []
    errors = []
    for ep in endpoints:
        if ep["status"] == "success":
            data = ep.get("data")
            record_count = None
            if isinstance(data, dict):
                record_count = data.get("fullResultSize", data.get("count"))
            success.append({
                "endpoint": ep["endpoint"],
                "params": ep.get("params"),
                "record_count": record_count,
            })
        else:
            errors.append({
                "endpoint": ep["endpoint"],
                "error": ep.get("error", "unknown"),
                "params": ep.get("params"),
            })
    return {"success": success, "errors": errors}


def print_summary(result: dict, verbose: bool = False):
    print(f"{'=' * 80}")
    print(f"Request: {result['request_id']}")
    print(f"Prompt:  {result['prompt']}")
    if result["files"]:
        print(f"Files:   {', '.join(result['files'])}")
    print(f"Time:    {result['timestamp_start']} → {result['timestamp_end']} ({result['duration_ms']:.0f}ms)")
    print(f"Status:  {result['status_code']}")
    print(f"Base URL:{result['base_url']}")
    print()

    summary = summarize_endpoints(result["endpoints"])

    print(f"  Endpoints hit: {len(result['endpoints'])} ({len(summary['success'])} success, {len(summary['errors'])} errors)")
    print()

    print("  SUCCESSFUL:")
    for ep in summary["success"]:
        params_str = f" params={ep['params']}" if ep["params"] else ""
        count_str = f" [{ep['record_count']} records]" if ep["record_count"] is not None else ""
        print(f"    ✓ {ep['endpoint']}{params_str}{count_str}")

    if summary["errors"]:
        print()
        print("  FAILED:")
        for ep in summary["errors"]:
            params_str = f" params={ep['params']}" if ep["params"] else ""
            err_short = ep["error"][:80] if ep["error"] else "unknown"
            print(f"    ✗ {ep['endpoint']}{params_str} — {err_short}")

    if verbose:
        print()
        print("  RESPONSE DATA:")
        for ep in result["endpoints"]:
            if ep["status"] == "success" and ep.get("data"):
                data = ep["data"]
                if isinstance(data, dict):
                    values = data.get("values", [])
                    if values:
                        print(f"\n    --- {ep['endpoint']} ({len(values)} records) ---")
                        print(f"    {json.dumps(values, indent=4, ensure_ascii=False)[:2000]}")
                    else:
                        print(f"\n    --- {ep['endpoint']} (empty) ---")

    print()


def analyze_all(log_dir: str, verbose: bool = False, limit: int | None = None):
    files = sorted(Path(log_dir).glob("*.json"))
    if limit:
        files = files[:limit]

    results = []
    for f in files:
        result = parse_log_file(str(f))
        results.append(result)

    for r in results:
        print_summary(r, verbose=verbose)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Tripletex request logs")
    parser.add_argument("--dir", default="data/parsed_logs_v2", help="Log directory")
    parser.add_argument("--file", help="Analyze a single log file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show response data")
    parser.add_argument("--limit", "-n", type=int, help="Limit number of files")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.file:
        result = parse_log_file(args.file)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_summary(result, verbose=args.verbose)
    else:
        results = analyze_all(args.dir, verbose=args.verbose, limit=args.limit)
        if args.json:
            # Strip large data fields for JSON output
            for r in results:
                for ep in r["endpoints"]:
                    if ep.get("data") and isinstance(ep["data"], dict):
                        values = ep["data"].get("values", [])
                        ep["data_record_count"] = len(values) if isinstance(values, list) else None
                        ep["data_summary"] = {
                            k: v for k, v in ep["data"].items() if k != "values"
                        }
                        del ep["data"]
            print(json.dumps(results, indent=2, ensure_ascii=False))
