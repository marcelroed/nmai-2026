"""End-to-end test for task 27 (currency invoice with exchange rate difference).

Environment assumptions tested here:
  - Customer "Fjorden AS" (org 999888777) has an outstanding EUR invoice (#36,
    5000 EUR booked at ~11.2955 NOK/EUR = 56477.50 NOK).
  - Accounts 8060 (agio), 8160 (disagio), 1500, 1920 exist.
  - Exactly one invoice payment type with debitAccount=1920.

Scoring assumptions tested here:
  - Payment registered at the new rate (paidAmount in NOK, paidAmountCurrency in EUR).
  - Exchange rate difference voucher posted to 8160 (disagio) or 8060 (agio).
  - Invoice amountOutstanding becomes 0 after payment.

Not tested:
  - Whether Tripletex auto-handles the FX difference (we post it manually;
    if auto-handled this would double-count).
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials, get_current_year_month_day_utc
from tripletex.parsers.task27_currency_invoice import CurrencyInvoice


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def _create_eur_invoice(creds: TripletexCredentials, customer_id: int, eur_amount: float) -> int:
    """Create an EUR invoice and return its ID."""
    today = get_current_year_month_day_utc()
    r = requests.post(
        f"{creds.base_url}/invoice",
        auth=creds.auth,
        json={
            "invoiceDate": today,
            "invoiceDueDate": get_current_year_month_day_utc(days_offset_forward=30),
            "customer": {"id": customer_id},
            "currency": {"id": 5},  # EUR
            "orders": [{
                "customer": {"id": customer_id},
                "orderDate": today,
                "deliveryDate": today,
                "currency": {"id": 5},
                "orderLines": [{
                    "count": 1,
                    "unitPriceExcludingVatCurrency": eur_amount,
                    "vatType": {"id": 0},
                    "description": "EUR test service",
                }],
            }],
        },
    )
    r.raise_for_status()
    return r.json()["value"]["id"]


def test_task27_solve_hardcoded():
    """Create an EUR invoice, then register payment at a lower rate (disagio)."""
    creds = _sandbox_creds()

    # Setup: create a fresh EUR invoice
    # Customer Fjorden AS (id=108134948, org=999888777)
    invoice_id = _create_eur_invoice(creds, customer_id=108134948, eur_amount=5000.0)

    # The invoice is booked at Tripletex's current EUR rate (~11.2955 NOK/EUR).
    # We'll pay at 10.50 → disagio.
    parsed = CurrencyInvoice(
        eur_amount=5000.0,
        customer_name="Fjorden AS",
        org_number="999888777",
        original_rate=11.2955,
        payment_rate=10.50,
        exchange_rate_type="disagio",
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: invoice is fully paid ────────────────────────────────────
    inv = requests.get(
        f"{creds.base_url}/invoice/{invoice_id}",
        auth=creds.auth,
        params={"fields": "id,amountOutstanding,amountCurrencyOutstanding"},
    ).json()["value"]
    assert inv["amountCurrencyOutstanding"] == 0.0, (
        f"EUR outstanding should be 0, got {inv['amountCurrencyOutstanding']}"
    )

    # ── Verify: FX difference voucher exists ─────────────────────────────
    vouchers = requests.get(
        f"{creds.base_url}/ledger/voucher",
        auth=creds.auth,
        params={
            "dateFrom": "2026-03-22",
            "dateTo": "2026-03-23",
            "fields": "id,description,postings(*)",
        },
    ).json()["values"]
    fx_vouchers = [v for v in vouchers if "disagio" in v["description"].lower()]
    assert len(fx_vouchers) >= 1, f"FX voucher not found. Got: {[v['description'] for v in vouchers]}"

    # Check the difference amount: 5000 * (11.2955 - 10.50) = 3977.50
    fx_postings = [p for p in fx_vouchers[-1]["postings"] if p["row"] > 0]
    amounts = sorted([p["amount"] for p in fx_postings])
    expected_diff = round(5000.0 * (11.2955 - 10.50), 2)
    assert amounts == [-expected_diff, expected_diff], (
        f"Expected amounts [-{expected_diff}, {expected_diff}], got {amounts}"
    )


def test_task27_parse():
    """Parse prompts to verify field extraction."""
    parsed = CurrencyInvoice.parse(
        "Enviámos uma fatura de 8676 EUR ao Cascata Lda (org. nº 967770892) quando a taxa de câmbio era 11.62 NOK/EUR. O cliente pagou agora, mas a taxa é 10.89 NOK/EUR. Registe o pagamento e lance a diferença cambial (disagio) na conta correta."
    )
    assert parsed.eur_amount == 8676.0
    assert parsed.org_number == "967770892"
    assert parsed.original_rate == 11.62
    assert parsed.payment_rate == 10.89
    assert parsed.exchange_rate_type == "disagio"
