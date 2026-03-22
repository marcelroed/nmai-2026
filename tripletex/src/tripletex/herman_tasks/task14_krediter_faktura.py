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

self_customer_org_num = "999000111"
self_tripletex_creds = TripletexCredentials.placeholder_TODO()
self_amount_excluding_vat = 18199.0

# +

customer = get_customer_by_org_number(
    org_number=self_customer_org_num, tripletex_creds=self_tripletex_creds
)

# +

invoice = get_invoice_by_amount_excluding_vat(
    amount_excluding_vat=self_amount_excluding_vat, tripletex_creds=self_tripletex_creds
)


# +

credit_note = requests.put(
    f"{self_tripletex_creds.base_url}/invoice/{invoice['id']}/:createCreditNote",
    auth=self_tripletex_creds.auth,
    params={
        "date": invoice["invoiceDate"]
    },  # TODO: i might want to not send to customer
).json()
credit_note
