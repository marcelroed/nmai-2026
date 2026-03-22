"""End-to-end test for task 23 (bank reconciliation) against the sandbox.

Environment assumptions (partially tested — sandbox lacks matching invoices):
  - Customer invoices exist with invoiceNumber matching CSV (e.g. 1001).
  - Supplier invoices exist, matchable by supplier name.
  - Exactly one invoice payment type with debitAccount=1920.
  - At least one outgoing payment type with creditAccount=1920.
  - Standard chart of accounts: 7770, 8040, 8150, 1920.

Scoring assumptions tested here (fee/interest voucher only):
  - Fee voucher: debit 7770, credit 1920 for the fee amount.
  - Interest income voucher: debit 1920, credit 8040 for interest amount.
  - Combined into a single voucher (4 postings).

Not tested (sandbox lacks pre-populated invoices):
  - Incoming payment registration (PUT /invoice/{id}/:payment).
  - Outgoing payment registration (POST /supplierInvoice/{id}/:addPayment).
  - Supplier invoice matching heuristic (name + amount).
  - Partial payment handling.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.parsers.task23_bank_reconciliation_csv import (
    BankFee,
    BankReconciliation,
    Interest,
)


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task23_fee_and_interest_voucher():
    """Test that bank fees and interest get posted as a single voucher.

    Skips incoming/outgoing payments since the sandbox doesn't have matching
    invoices for the CSV data. Those are just PUT/POST calls that follow the
    same pattern as task 7.
    """
    creds = _sandbox_creds()

    parsed = BankReconciliation(
        incoming_payments=[],
        outgoing_payments=[],
        bank_fees=[BankFee(date="2026-01-29", amount=482.20)],
        interest=[Interest(date="2026-01-30", amount=927.54, is_income=True)],
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: voucher was created ──────────────────────────────────────
    vouchers = requests.get(
        f"{creds.base_url}/ledger/voucher",
        auth=creds.auth,
        params={
            "dateFrom": "2026-01-29",
            "dateTo": "2026-01-31",
            "fields": "id,date,description,postings(*)",
        },
    ).json()["values"]

    match = [v for v in vouchers if v["description"] == "Bankgebyr og renter"]
    assert len(match) >= 1, f"Fee/interest voucher not found. Got: {vouchers}"
    voucher = match[-1]

    # Should have 4 postings: fee debit+credit, interest debit+credit
    postings = [p for p in voucher["postings"] if p["row"] > 0]
    assert len(postings) == 4, f"Expected 4 postings, got {len(postings)}: {postings}"


def test_task23_parse():
    """Parse a real bank statement CSV with the LLM."""
    from pathlib import Path

    attachment = Path("data/files/raw/bankutskrift_en_05.csv").read_text()
    prompt = "Reconcile the bank statement (attached CSV) against open invoices in Tripletex. Match incoming payments to customer invoices and outgoing payments to supplier invoices. Handle partial payments correctly."

    parsed = BankReconciliation.parse(prompt, attachment)

    assert len(parsed.incoming_payments) == 5
    assert len(parsed.outgoing_payments) == 3
    assert len(parsed.bank_fees) == 1
    assert len(parsed.interest) == 1

    # Check first incoming payment
    assert parsed.incoming_payments[0].customer_name == "Wilson Ltd"
    assert parsed.incoming_payments[0].invoice_number == 1001
    assert parsed.incoming_payments[0].amount == 27750.0

    # Check bank fee
    assert parsed.bank_fees[0].amount == 482.20

    # Check interest
    assert parsed.interest[0].amount == 927.54
    assert parsed.interest[0].is_income is True
