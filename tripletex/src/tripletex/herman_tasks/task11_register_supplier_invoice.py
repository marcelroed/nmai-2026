from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_customer_by_org_number,
    get_products_by_product_numbers,
)
from tripletex.my_log import configure_logging

logger = configure_logging()
# +

self_invoice_name = "INV-2026-6293"
self_supplier_org_num = "999000111"
self_send_to_customer = False
self_tripletex_creds = TripletexCredentials.placeholder_TODO()
