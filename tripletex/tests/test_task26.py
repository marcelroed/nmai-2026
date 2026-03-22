"""End-to-end test for task 26 (monthly closing) against the sandbox.

Environment assumptions tested here:
  - Standard chart of accounts present (1700, 6300, 6020, 1080, 5000, 2900, 1920).
  - Employee employment details exist with monthlySalary > 0.

Scoring assumptions tested here:
  - A single voucher with 6 postings (accrual, depreciation, salary provision).
  - Accrual: debit expense account, credit prepaid account.
  - Depreciation: debit depreciation account, credit contra-asset.
  - Salary: debit 5000, credit 2900 for total monthly salary.

Not tested:
  - Whether contra-asset mapping (DEPRECIATION_CONTRA_MAP) matches scorer expectations.
  - Whether salary provision should include employer contributions beyond base salary.
  - Whether scorer expects 3 separate vouchers instead of 1 combined.
  - Account 6030 (used in some prompts) doesn't exist in our sandbox.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.parsers.task26_monthly_closing import MonthlyClosing


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task26_solve_hardcoded():
    """Run monthly closing with hardcoded values."""
    creds = _sandbox_creds()

    # Use account 6020 (exists) instead of 6030 (doesn't exist in sandbox)
    parsed = MonthlyClosing(
        accrual_amount_per_month=4200.0,
        accrual_source_account=1700,
        asset_cost=64000.0,
        useful_life_years=3,
        depreciation_account=6020,
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: voucher was created with 6 postings ──────────────────────
    vouchers = requests.get(
        f"{creds.base_url}/ledger/voucher",
        auth=creds.auth,
        params={
            "dateFrom": "2026-03-31",
            "dateTo": "2026-04-01",
            "fields": "id,description,postings(*)",
        },
    ).json()["values"]

    match = [v for v in vouchers if v["description"] == "Månedsslutt mars 2026"]
    assert len(match) >= 1, f"Monthly closing voucher not found. Got: {vouchers}"
    voucher = match[-1]

    postings = [p for p in voucher["postings"] if p["row"] > 0]
    assert len(postings) == 6, f"Expected 6 postings, got {len(postings)}"

    # Check accrual: row 1 = debit 6300 (4200), row 2 = credit 1700 (-4200)
    assert postings[0]["amount"] == 4200.0
    assert postings[1]["amount"] == -4200.0

    # Check depreciation: row 3 = debit 6020, row 4 = credit 1080
    dep_monthly = round(64000.0 / 3 / 12, 2)
    assert postings[2]["amount"] == dep_monthly
    assert postings[3]["amount"] == -dep_monthly

    # Check salary provision: row 5 = debit 5000, row 6 = credit 2900
    assert postings[4]["amount"] > 0  # salary amount from API
    assert postings[5]["amount"] < 0
    assert postings[4]["amount"] == -postings[5]["amount"]


def test_task26_parse():
    """Parse a prompt to verify field extraction."""
    prompt = "Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (4200 NOK pro Monat von Konto 1700 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 64000 NOK und Nutzungsdauer 3 Jahre (lineare Abschreibung auf Konto 6030). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900)."

    parsed = MonthlyClosing.parse(prompt)

    assert parsed.accrual_amount_per_month == 4200.0
    assert parsed.accrual_source_account == 1700
    assert parsed.asset_cost == 64000.0
    assert parsed.useful_life_years == 3
    assert parsed.depreciation_account == 6030
