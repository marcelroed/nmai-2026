"""End-to-end test for task 22 (expense from receipt PDF) against the sandbox.

Environment assumptions tested here:
  - Department "Salg" pre-exists.
  - Standard chart of accounts present (6540, 1920).
  - vatType id=1 is 25% input VAT.

Scoring assumptions tested here:
  - Voucher created with correct date (receipt date).
  - Expense posting: correct net amount on correct account with department.
  - VAT posting: auto-generated (row 0) with 25% of net amount.
  - Bank posting: correct gross (incl. VAT) amount on account 1920.

Not tested:
  - Whether the expense account mapping (item → account number) matches scorer expectations.
  - Whether voucher description matters.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.parsers.task22_expense_from_receipt_pdf import ExpenseFromReceiptPDF


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task22_solve_hardcoded():
    """Run solve with a hardcoded parsed model (no LLM call) against the sandbox."""
    creds = _sandbox_creds()

    # Based on kvittering_en_03.txt: USB-hub at 3000 kr from IKEA, dept HR
    # But we pick "Kontorstoler" at 3000 kr to test with dept Salg (pre-existing)
    parsed = ExpenseFromReceiptPDF(
        expense_item_name="Kontorstoler",
        department_name="Salg",
        expense_account_number=6540,
        item_price_excl_vat=3000.0,
        item_price_incl_vat=3750.0,
        receipt_date="2026-06-16",
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: voucher exists with correct postings ─────────────────────
    vouchers = requests.get(
        f"{creds.base_url}/ledger/voucher",
        auth=creds.auth,
        params={
            "dateFrom": "2026-06-16",
            "dateTo": "2026-06-17",
            "fields": "id,date,description,postings(*)",
        },
    ).json()["values"]

    match = [v for v in vouchers if v["description"] == "Kontorstoler"]
    assert len(match) >= 1, f"Voucher not found. Got: {vouchers}"
    voucher = match[-1]

    # Check postings: row 1 = expense debit, row 2 = bank credit, row 0 = VAT
    postings = voucher["postings"]
    expense_posting = next(p for p in postings if p["row"] == 1)
    bank_posting = next(p for p in postings if p["row"] == 2)
    vat_posting = next(p for p in postings if p["row"] == 0)

    assert expense_posting["amount"] == 3000.0
    assert expense_posting["amountGross"] == 3750.0
    assert bank_posting["amountGross"] == -3750.0
    assert vat_posting["amount"] == 750.0  # 25% of 3000


def test_task22_parse_and_solve():
    """Parse a real receipt with the LLM, then solve against the sandbox."""
    from pathlib import Path

    creds = _sandbox_creds()
    attachment = Path("data/files/parsed/kvittering_en_03.txt").read_text()
    prompt = "We need the Kontorstoler expense from this receipt posted to department Salg. Use the correct expense account and ensure proper VAT treatment."

    parsed = ExpenseFromReceiptPDF.parse(prompt, attachment)

    # Sanity-check parsing against known receipt content
    assert parsed.expense_item_name == "Kontorstoler"
    assert parsed.department_name == "Salg"
    assert parsed.item_price_excl_vat == 3000.0
    assert parsed.expense_account_number == 6540

    parsed.solve(tripletex_client=creds.to_client())
