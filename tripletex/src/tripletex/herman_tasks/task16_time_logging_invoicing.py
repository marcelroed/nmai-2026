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
self_hours: int = 18
self_first_name: str = "Sophia"
self_last_name: str = "Schmidt"
self_email: str = "sophia.schmidt@example.org"
self_activity_name: str = "Design"
self_project_name: str = "Sicherheitsaudit"
self_customer_name: str = "Windkraft GmbH"
self_org_number: str = "882984826"
self_hourly_rate: float = 950.0
# +
