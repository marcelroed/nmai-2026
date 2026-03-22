from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_customer_by_org_number,
    get_products_by_product_numbers,
)
from tripletex.my_log import configure_logging

logger = configure_logging()
# +

# self_org_num = "900314183"
self_org_num = "999000111"
self_product_numbers = ["4646", "8115-2"]
self_send_to_customer = False
self_tripletex_creds = TripletexCredentials.placeholder_TODO()

# +

customer = get_customer_by_org_number(
    org_number=self_org_num, tripletex_creds=self_tripletex_creds
)

# +

products = get_products_by_product_numbers(
    product_numbers=self_product_numbers, tripletex_creds=self_tripletex_creds
)

# TODO: understand if we need to create the orders before we create the invoice
# +
