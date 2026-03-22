import requests

from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_current_year_month_day_utc,
    get_customer_by_org_number,
    get_product_by_product_number,
    get_products_by_product_numbers,
)
from tripletex.log_analysis._3_find_by_task import get_task
from tripletex.my_log import configure_logging

logger = configure_logging()
# +

# self_org_num = "900314183"
self_org_num = "999000111"
self_product_numbers = ["4646", "8115-2"]
# self_send_to_customer = True
self_tripletex_creds = TripletexCredentials.placeholder_TODO()


# +

customer = get_customer_by_org_number(
    org_number=self_org_num, tripletex_creds=self_tripletex_creds
)
customer

# +

products = get_products_by_product_numbers(
    product_numbers=self_product_numbers, tripletex_creds=self_tripletex_creds
)

# +

json_payload = {
    "invoiceDate": get_current_year_month_day_utc(),
    "invoiceDueDate": get_current_year_month_day_utc(days_offset_forward=30),
    "customer": {"id": customer["id"]},
    "orders": [
        {
            "customer": {"id": customer["id"]},
            "orderDate": get_current_year_month_day_utc(),
            "orderLineSorting": "CUSTOM",
            "deliveryDate": get_current_year_month_day_utc(),
            "orderLines": [
                {
                    "count": 1,
                    "unitPriceExcludingVatCurrency": product[
                        "priceExcludingVatCurrency"
                    ],
                    "vatType": product["vatType"],
                    "sortIndex": i,
                    "description": product["displayName"],
                }
                for i, product in enumerate(products)
            ],
        }
    ],
}
json_payload

# +

logger.info(json_payload)
r = requests.post(
    f"{self_tripletex_creds.base_url}/invoice",
    auth=self_tripletex_creds.auth,
    # params={"sendToCustomer": self_send_to_customer},
    json=json_payload,
)
r.json()
