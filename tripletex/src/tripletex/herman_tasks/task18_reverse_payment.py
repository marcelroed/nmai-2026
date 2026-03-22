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

# +

self_tripletex_creds = TripletexCredentials.placeholder_TODO()
self_customer_name: str = "Ridgepoint Ltd"
self_org_number: str = "999000111"
self_invoice_description: str = "Cloud Storage"
self_amount_excl_vat: float = 18199

# +

customer = get_customer_by_org_number(
    org_number=self_org_number, tripletex_creds=self_tripletex_creds
)

# +

invoice = get_invoice_by_amount_excluding_vat(
    amount_excluding_vat=self_amount_excl_vat, tripletex_creds=self_tripletex_creds
)
invoice

# +

logger.info(f"{invoice["customer"]["id"] == customer["id"]=}")
invoice["postings"]

# +

postings = requests.get(
    f"{self_tripletex_creds.base_url}/ledger/posting",
    auth=self_tripletex_creds.auth,
    params={"dateFrom": "1970-01-01", "dateTo": "2027-12-31"},
).json()
postings

# +

invoice_posting_ids = [posting["id"] for posting in invoice["postings"]]
invoice_posting_ids
(correct_posting,) = [
    x
    for x in postings["values"]
    if x["id"] in invoice_posting_ids
    and x["type"] == "OUTGOING_INVOICE_CUSTOMER_POSTING"
]

# +


requests.put(
    f"{self_tripletex_creds}/ledger/voucher/{invoice['voucher']['id']}/:reverse",
    auth=self_tripletex_creds.auth,
    params={"date": correct_posting["date"]},
)
