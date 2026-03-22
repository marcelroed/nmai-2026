"""End-to-end test for task 17 (custom dimension + voucher).

Environment assumptions:
  - Max 3 custom dimensions allowed. Clean sandbox has 0.
  - Our persistent sandbox already has 3 — cannot test dimension creation.

Scoring assumptions:
  - Dimension created with correct name.
  - Both dimension values created.
  - Voucher posted on correct account, correct amount, linked to the
    specified dimension value via freeAccountingDimension{N}.

Tested here:
  - Parse extracts all fields correctly.
  - Voucher creation with dimension link works (using existing dimension).
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials, get_current_year_month_day_utc
from tripletex.parsers.task17_custom_dimension import CustomDimension


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task17_voucher_with_dimension():
    """Test voucher posting linked to an existing dimension value.

    Cannot test full flow (dimension creation) because sandbox already has 3 dimensions.
    Uses existing dimension 'Produktlinje' (index=1), value 'Avansert' (id=16131).
    """
    creds = _sandbox_creds()
    today = get_current_year_month_day_utc()

    accts_r = requests.get(
        f"{creds.base_url}/ledger/account",
        auth=creds.auth,
        params={"fields": "id,number", "count": 1000},
    )
    accts = {a["number"]: a for a in accts_r.json()["values"]}

    r = requests.post(
        f"{creds.base_url}/ledger/voucher",
        auth=creds.auth,
        json={
            "date": today,
            "description": "task17 test",
            "postings": [
                {
                    "account": {"id": accts[6340]["id"]},
                    "amountCurrency": 15000,
                    "amountGross": 15000,
                    "amountGrossCurrency": 15000,
                    "freeAccountingDimension1": {"id": 16131},  # Avansert
                    "row": 1,
                },
                {
                    "account": {"id": accts[1920]["id"]},
                    "amountCurrency": -15000,
                    "amountGross": -15000,
                    "amountGrossCurrency": -15000,
                    "row": 2,
                },
            ],
        },
    )
    r.raise_for_status()
    posting = r.json()["value"]["postings"][0]
    assert posting["freeAccountingDimension1"]["id"] == 16131


def test_task17_parse():
    parsed = CustomDimension.parse(
        'Opprett ein fri rekneskapsdimensjon "Produktlinje" med verdiane "Basis" og "Avansert". Bokfør deretter eit bilag på konto 6340 for 15000 kr, knytt til dimensjonsverdien "Avansert".'
    )
    assert parsed.dimension_name == "Produktlinje"
    assert set(parsed.dimension_values) == {"Basis", "Avansert"}
    assert parsed.account_number == 6340
    assert parsed.amount == 15000.0
    assert parsed.linked_dimension_value == "Avansert"
