"""Test harness: runs all competition queries against local endpoint and reports API call counts."""
import json
import subprocess
import sys
import time

QUERIES = json.loads(open("queries.json").read())

BASE_URL = "http://localhost:8000/solve"
TOKEN = ""  # removed
TRIPLETEX_BASE = "https://kkpqfuj-amager.tripletex.dev/v2"


def run_query(prompt: str, task_type: str):
    """Send a query to the local endpoint and return timing."""
    payload = json.dumps({
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": TRIPLETEX_BASE,
            "session_token": TOKEN,
        }
    })

    start = time.monotonic()
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", BASE_URL,
         "-H", "Content-Type: application/json",
         "-d", payload],
        capture_output=True, text=True, timeout=30
    )
    elapsed = time.monotonic() - start

    response = result.stdout.strip()
    return {
        "type": task_type,
        "time": round(elapsed, 2),
        "response": response,
        "ok": '"completed"' in response,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("COMPETITION QUERY TEST")
    print("=" * 60)

    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] {q['type']}...")
        r = run_query(q["prompt"], q["type"])
        results.append(r)
        status = "✅" if r["ok"] else "❌"
        print(f"  {status} {r['time']}s — {r['response'][:50]}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok = sum(1 for r in results if r["ok"])
    print(f"Passed: {ok}/{len(results)}")
    print(f"Total time: {sum(r['time'] for r in results):.1f}s")
    print(f"Avg time: {sum(r['time'] for r in results)/len(results):.1f}s")
    print()
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"  {status} {r['type']:40s} {r['time']:5.1f}s")
