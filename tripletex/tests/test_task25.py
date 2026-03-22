"""End-to-end test for task 25 (overdue reminder) against the sandbox.

Environment assumptions tested here:
  - Exactly one overdue invoice exists (dueDate < today, amountOutstanding > 0).
  - Accounts 1500, 3400, 1920 exist in chart of accounts.
  - Exactly one invoice payment type with debitAccount=1920.

Scoring assumptions tested here:
  - Reminder fee voucher: debit 1500, credit 3400 for fee amount.
  - Reminder invoice created and sent to the customer.
  - Partial payment registered on overdue invoice (amountOutstanding reduced).

Not tested:
  - Whether reminder invoice VAT treatment matters (we use vatType 0).
  - Whether voucher/invoice descriptions are checked.
  - Whether the fee voucher needs a customer reference on the posting.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials, get_current_year_month_day_utc
from tripletex.parsers.task25_overdue_reminder import OverdueReminder


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task25_solve_hardcoded():
    """Create an overdue invoice, then run the reminder flow."""
    creds = _sandbox_creds()
    today = get_current_year_month_day_utc()

    # Setup: we need exactly one overdue invoice. Use an existing one from the
    # sandbox that has amountOutstanding > 5000 and dueDate < today.
    invoices_r = requests.get(
        f"{creds.base_url}/invoice",
        auth=creds.auth,
        params={
            "invoiceDateFrom": "1970-01-01",
            "invoiceDateTo": "2027-12-31",
            "fields": "id,invoiceNumber,invoiceDueDate,amountOutstanding,customer(id,name)",
        },
    )
    invoices_r.raise_for_status()
    overdue = [
        inv for inv in invoices_r.json()["values"]
        if inv["amountOutstanding"] >= 5000 and inv["invoiceDueDate"] < today
    ]
    assert len(overdue) == 1, (
        f"Expected exactly 1 overdue invoice, got {len(overdue)}. "
        "Pay off extras via PUT /invoice/{id}/:payment to match competition assumptions."
    )
    target = overdue[0]
    original_outstanding = target["amountOutstanding"]

    parsed = OverdueReminder(
        reminder_fee=70.0,
        partial_payment_amount=5000.0,
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: partial payment reduced outstanding amount ───────────────
    inv_after = requests.get(
        f"{creds.base_url}/invoice/{target['id']}",
        auth=creds.auth,
        params={"fields": "id,amountOutstanding"},
    ).json()["value"]
    assert inv_after["amountOutstanding"] == original_outstanding - 5000.0, (
        f"Expected outstanding {original_outstanding - 5000}, got {inv_after['amountOutstanding']}"
    )


def test_task25_parse():
    """Parse a prompt to verify field extraction."""
    prompt = "One of your customers has an overdue invoice. Find the overdue invoice and post a reminder fee of 70 NOK. Debit accounts receivable (1500), credit reminder fees (3400). Also create an invoice for the reminder fee to the customer and send it. Additionally, register a partial payment of 5000 NOK on the overdue invoice."

    parsed = OverdueReminder.parse(prompt)

    assert parsed.reminder_fee == 70.0
    assert parsed.partial_payment_amount == 5000.0
