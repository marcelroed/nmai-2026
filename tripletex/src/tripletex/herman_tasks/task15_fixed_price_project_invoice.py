from decimal import ROUND_HALF_UP, Decimal

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
self_fixed_price: float = 12345
self_project_name: str = "Run46 Hours"
self_customer_name: str = "Sonnental GmbH"
self_org_number: str = "999000111"
self_project_manager_first_name: str = "Finn"
self_project_manager_last_name: str = "Müller"
self_project_manager_email: str = "fernando.garcia@example.org"
self_invoice_percentage: float = 0.25

# +

customer = get_customer_by_org_number(
    org_number=self_org_number, tripletex_creds=self_tripletex_creds
)

# +

project_manager_employee = get_employee(
    self_project_manager_email, tripletex_creds=self_tripletex_creds
)
project_manager_employee

# +

projects = requests.get(
    f"{self_tripletex_creds.base_url}/project", auth=self_tripletex_creds.auth
).json()["values"]

# +

(project,) = [
    project for project in projects if self_project_name in project["displayName"]
]
project

# +

project

# +

project_payload = {
    "version": 0,
    # "projectManager": {"id": project_manager_employee["id"]},
    "isFixedPrice": True,
    "fixedprice": self_fixed_price,
}
project_payload

# +

update_project_r = requests.put(
    f"{self_tripletex_creds.base_url}/project/{project['id']}",
    auth=self_tripletex_creds.auth,
    json=project_payload,
).json()
update_project_r["value"]

# +

order_payload = {
    "customer": {"id": customer["id"]},
    "project": {"id": project["id"]},
    "orderDate": get_current_year_month_day_utc(),
    "deliveryDate": get_current_year_month_day_utc(),
}
order_payload

# +

order_r = requests.post(
    f"{self_tripletex_creds.base_url}/order",
    auth=self_tripletex_creds.auth,
    json=order_payload,
).json()
order_r["value"]["id"]

# +

amount_on_account = (
    Decimal(self_fixed_price) * Decimal(self_invoice_percentage) / Decimal("100")
).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

invoice_params = {
    "invoiceDate": get_current_year_month_day_utc(),
    "createOnAccount": "WITHOUT_VAT",
    "amountOnAccount": str(amount_on_account),
    "onAccountComment": "Milestone payment",
}
invoice_params

# +

invoice_r = requests.put(
    f"{self_tripletex_creds.base_url}/order/{order_r['value']['id']}/:invoice",
    params=invoice_params,
    auth=self_tripletex_creds.auth,
).json()
invoice_r
