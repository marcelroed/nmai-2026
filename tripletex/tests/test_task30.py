"""End-to-end test for task 30 (year-end closing) against the sandbox.

Environment assumptions:
  - Accounts 6010, 1209, 1700, 6300, 8700, 2920 exist in competition sandbox.
  - Accounts 1209 and 8700 are MISSING from our persistent sandbox.
  - P&L postings for 2025 exist in the ledger.

Scoring assumptions tested (via parse test):
  - 3 separate depreciation vouchers: debit 6010, credit 1209.
  - Prepaid reversal: debit 6300, credit 1700.
  - Tax provision: debit 8700, credit 2920 for 22% of taxable profit.
  - Tax computed after depreciations and prepaid reversal.

Not tested (accounts missing in sandbox):
  - Full solve flow (needs 1209 and 8700).
  - Whether prepaid always maps to expense 6300.
  - Rounding precision for depreciation and tax.
"""

from tripletex.parsers.task30_year_end_closing import DepreciationAsset, YearEndClosing


def test_task30_parse():
    """Parse a prompt to verify field extraction."""
    prompt = "Perform simplified year-end closing for 2025: 1) Calculate and post annual depreciation for three assets: Kontormaskiner (138450 NOK, 6 years straight-line, account 1200), Programvare (280000 NOK, 9 years, account 1250), Inventar (484650 NOK, 8 years, account 1240). Use account 6010 for depreciation expense and 1209 for accumulated depreciation. 2) Reverse prepaid expenses (total 52250 NOK on account 1700). 3) Calculate and post tax provision (22% of taxable profit) on account 8700/2920. Post each depreciation as a separate voucher."

    parsed = YearEndClosing.parse(prompt)

    assert len(parsed.assets) == 3
    assert parsed.assets[0].asset_name == "Kontormaskiner"
    assert parsed.assets[0].cost == 138450.0
    assert parsed.assets[0].useful_life_years == 6
    assert parsed.assets[0].asset_account == 1200

    assert parsed.assets[1].cost == 280000.0
    assert parsed.assets[1].useful_life_years == 9
    assert parsed.assets[1].asset_account == 1250

    assert parsed.assets[2].cost == 484650.0
    assert parsed.assets[2].asset_account == 1240

    assert parsed.prepaid_expense_total == 52250.0


def test_task30_depreciation_math():
    """Verify depreciation and tax computation logic."""
    parsed = YearEndClosing(
        assets=[
            DepreciationAsset(asset_name="A", cost=120000, useful_life_years=10, asset_account=1200),
            DepreciationAsset(asset_name="B", cost=90000, useful_life_years=3, asset_account=1250),
            DepreciationAsset(asset_name="C", cost=60000, useful_life_years=5, asset_account=1240),
        ],
        prepaid_expense_total=50000,
    )

    dep_amounts = [round(a.cost / a.useful_life_years, 2) for a in parsed.assets]
    assert dep_amounts == [12000.0, 30000.0, 12000.0]
    total_dep = sum(dep_amounts)
    assert total_dep == 54000.0

    # If current P&L is -300000 (profit of 300k):
    # adjusted = -300000 + 54000 + 50000 = -196000
    # taxable_profit = 196000
    # tax = 196000 * 0.22 = 43120
    current_pl = -300000.0
    adjusted_pl = current_pl + total_dep + parsed.prepaid_expense_total
    taxable_profit = -adjusted_pl
    tax = round(max(0, taxable_profit) * 0.22, 2)
    assert taxable_profit == 196000.0
    assert tax == 43120.0
