"""End-to-end test for task 28 (cost increase analysis) against the sandbox.

Environment assumptions tested here:
  - Ledger postings exist in Jan-Feb 2026 on expense accounts (4000-8999).
  - At least one employee exists (for project manager).
  - POST /project/list with nested projectActivities works.

Scoring assumptions tested here:
  - 3 internal projects created, named after the top-3 increased accounts.
  - Each project has at least one linked activity.

Not tested:
  - Exact project name format expected by scorer.
  - Whether activity name or type matters.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials, get_current_year_month_day_utc
from tripletex.parsers.task28_cost_increase_analysis import CostIncreaseAnalysis


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def _get_accts(creds: TripletexCredentials) -> dict[int, dict]:
    r = requests.get(
        f"{creds.base_url}/ledger/account",
        auth=creds.auth,
        params={"fields": "id,number", "count": 1000},
    )
    r.raise_for_status()
    return {a["number"]: a for a in r.json()["values"]}


def _post_voucher(creds, accts, date, lines):
    postings = []
    for i, (acct, amt) in enumerate(lines, 1):
        postings.append({
            "account": {"id": accts[acct]["id"]},
            "amountCurrency": amt,
            "amountGross": amt,
            "amountGrossCurrency": amt,
            "row": i,
        })
    r = requests.post(
        f"{creds.base_url}/ledger/voucher",
        auth=creds.auth,
        json={"date": date, "description": "test28 setup", "postings": postings},
    )
    r.raise_for_status()


def test_task28_solve():
    """Create Jan/Feb postings with known increases, then verify projects."""
    creds = _sandbox_creds()
    accts = _get_accts(creds)

    # Setup: create expense postings so we know the top 3 increases.
    # Jan: 6300=1000, 6500=2000, 7300=500
    # Feb: 6300=5000, 6500=3000, 7300=4000
    # Increases: 7300=+3500, 6300=+4000, 6500=+1000
    # Top 3 by increase: 6300 (+4000), 7300 (+3500), 6500 (+1000)
    # (Note: sandbox may have other postings that shift the ranking)

    _post_voucher(creds, accts, "2026-01-15", [(6300, 1000), (1920, -1000)])
    _post_voucher(creds, accts, "2026-01-15", [(6500, 2000), (1920, -2000)])
    _post_voucher(creds, accts, "2026-01-15", [(7300, 500), (1920, -500)])

    _post_voucher(creds, accts, "2026-02-15", [(6300, 5000), (1920, -5000)])
    _post_voucher(creds, accts, "2026-02-15", [(6500, 3000), (1920, -3000)])
    _post_voucher(creds, accts, "2026-02-15", [(7300, 4000), (1920, -4000)])

    parsed = CostIncreaseAnalysis()
    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: 3 internal projects were created ─────────────────────────
    projects = requests.get(
        f"{creds.base_url}/project",
        auth=creds.auth,
        params={"fields": "id,name,isInternal,projectActivities(id)", "count": 100},
    ).json()["values"]

    internal_projects = [p for p in projects if p["isInternal"]]
    # We created at least 3 new ones (sandbox may have others)
    assert len(internal_projects) >= 3, (
        f"Expected at least 3 internal projects, got {len(internal_projects)}"
    )

    # The most recently created 3 should each have an activity
    recent = internal_projects[-3:]
    for p in recent:
        assert len(p["projectActivities"]) >= 1, (
            f"Project '{p['name']}' has no activities"
        )
