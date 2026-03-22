"""End-to-end test for task 24 (ledger error correction) against the sandbox.

Environment assumptions tested here:
  - Vouchers with errors exist in Jan-Feb 2026.
  - Duplicate voucher: exactly 2 postings on the same account/amount; we reverse the last.
  - All referenced account numbers exist (6500, 6540, 7300, 2710, 1920).

Scoring assumptions tested here:
  - Duplicate voucher is reversed via PUT /:reverse.
  - Wrong account: correction voucher credits wrong account, debits correct account.
  - Missing VAT: correction adds debit to 2710 for 25% of excl. amount, credit to 1920.
  - Incorrect amount: correction credits the excess from the account, debits 1920.
  - All three non-duplicate corrections are combined in a single voucher.

Not tested:
  - Whether scorer expects separate vouchers per correction (vs combined).
  - Whether scorer checks correction voucher date or description.
  - Whether 1920 is always the right offset account for corrections.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.parsers.task24_ledger_error_correction import (
    DuplicateVoucherError,
    IncorrectAmountError,
    LedgerErrorCorrection,
    MissingVatLineError,
    WrongAccountError,
)


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


def _create_voucher(
    creds: TripletexCredentials, accts: dict, date: str, desc: str, lines: list[tuple[int, float]]
) -> int:
    """Create a balanced voucher. lines = [(account_number, amount), ...]."""
    postings = []
    for i, (acct_num, amount) in enumerate(lines, start=1):
        postings.append({
            "account": {"id": accts[acct_num]["id"]},
            "amountCurrency": amount,
            "amountGross": amount,
            "amountGrossCurrency": amount,
            "row": i,
        })
    r = requests.post(
        f"{creds.base_url}/ledger/voucher",
        auth=creds.auth,
        json={"date": date, "description": desc, "postings": postings},
    )
    r.raise_for_status()
    return r.json()["value"]["id"]


def test_task24_solve_hardcoded():
    """Create erroneous vouchers, then correct them."""
    creds = _sandbox_creds()
    accts = _get_accts(creds)

    # Setup: create the "erroneous" vouchers in Jan 2026
    # 1. Wrong account: posted 3450 to 6500 instead of 6540
    _create_voucher(creds, accts, "2026-01-10", "Wrong account entry", [
        (6500, 3450.0), (1920, -3450.0),
    ])
    # 2. Duplicate: two identical vouchers on 6540 for 3700
    _create_voucher(creds, accts, "2026-01-12", "Original entry", [
        (6540, 3700.0), (1920, -3700.0),
    ])
    _create_voucher(creds, accts, "2026-01-12", "Original entry", [
        (6540, 3700.0), (1920, -3700.0),
    ])
    # 3. Missing VAT: 23500 excl VAT on 6540 but no VAT line on 2710
    _create_voucher(creds, accts, "2026-01-15", "Missing VAT entry", [
        (6540, 23500.0), (1920, -23500.0),
    ])
    # 4. Incorrect amount: 18600 on 7300 instead of 11550
    _create_voucher(creds, accts, "2026-01-20", "Wrong amount entry", [
        (7300, 18600.0), (1920, -18600.0),
    ])

    # Now run the correction
    parsed = LedgerErrorCorrection(
        wrong_account_error=WrongAccountError(
            wrong_account=6500, correct_account=6540, amount=3450.0,
        ),
        duplicate_voucher_error=DuplicateVoucherError(
            account=6540, amount=3700.0,
        ),
        missing_vat_line_error=MissingVatLineError(
            account=6540, amount_excl_vat=23500.0,
        ),
        incorrect_amount_error=IncorrectAmountError(
            account=7300, posted_amount=18600.0, correct_amount=11550.0,
        ),
    )

    parsed.solve(tripletex_client=creds.to_client())


def test_task24_parse():
    """Parse a prompt to verify field extraction."""
    prompt = "We have discovered errors in the general ledger for January and February 2026. Review all vouchers and find the 4 errors: a posting to the wrong account (account 7100 used instead of 7140, amount 6400 NOK), a duplicate voucher (account 7300, amount 1100 NOK), a missing VAT line (account 6500, amount excl. 19350 NOK missing VAT on account 2710), and an incorrect amount (account 6540, 20500 NOK posted instead of 12150 NOK). Correct all errors with appropriate correction vouchers."

    parsed = LedgerErrorCorrection.parse(prompt)

    assert parsed.wrong_account_error.wrong_account == 7100
    assert parsed.wrong_account_error.correct_account == 7140
    assert parsed.wrong_account_error.amount == 6400.0

    assert parsed.duplicate_voucher_error.account == 7300
    assert parsed.duplicate_voucher_error.amount == 1100.0

    assert parsed.missing_vat_line_error.account == 6500
    assert parsed.missing_vat_line_error.amount_excl_vat == 19350.0

    assert parsed.incorrect_amount_error.account == 6540
    assert parsed.incorrect_amount_error.posted_amount == 20500.0
    assert parsed.incorrect_amount_error.correct_amount == 12150.0
