import requests

from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_current_year_month_day_utc,
    get_customer_by_org_number,
    get_employee,
    get_invoice_by_amount_excluding_vat,
)
from tripletex.my_log import configure_logging

logger = configure_logging()

self_tripletex_creds = TripletexCredentials.placeholder_TODO()
self_dimension_name: str = "Prosjekttype"
self_dimension_values: list[str] = ["Utvikling", "Internt"]
self_account_number: str = "8945"
self_amount: float = 40600.0
self_linked_dimension_value: str = "Utvikling"

# +

ledger_accounts = requests.get(
    f"{self_tripletex_creds.base_url}/ledger/account", auth=self_tripletex_creds.auth
).json()

# +

(ledger_account,) = [
    ledger_account
    for ledger_account in ledger_accounts["values"]
    if str(ledger_account["number"]) == str(self_account_number)
]
ledger_account

# +

ledger_account

# +

requests.get(
    f"{self_tripletex_creds.base_url}/ledger/accountingDimensionName",
    auth=self_tripletex_creds.auth,
).json()
