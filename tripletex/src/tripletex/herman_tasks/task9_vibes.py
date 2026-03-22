import requests

from tripletex.config import AUTH, BASE_URL
from tripletex.log_analysis._3_find_by_task import get_task

tasks, raw_data = get_task("Task 9", only_log_version_2=True)

# +

task = tasks[0]
log_data = raw_data[task.request_id][::-1]

# +

task.prompt
product_numbers = ["9716", "6906", "2265"]

# +

[(i, x["message"]) for i, x in enumerate(log_data)]

# +

[(i, x["message"]) for i, x in enumerate(log_data) if "/customer" in x["message"]]

# +

(all_customers,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /customer")
]

# +

(customer,) = [
    x
    for x in all_customers["extra"]["data"]["values"]
    if x["organizationNumber"] == self_org_num
]
customer

# +


(all_products,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /product")
]
[x for x in all_products["extra"]["data"]["values"] if x["number"] in product_numbers]
# +

invoices = requests.get(
    f"{self_base_url}/invoice",
    auth=AUTH,
    params={"invoiceDateFrom": "1970-01-01", "invoiceDateTo": "2027-12-31"},
).json()
invoices["values"]

# +

(all_invoices,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /invoice")
]
all_invoices

# +

all_products["extra"]["data"]["values"][0]

# +


i = 0
product = products[0]
# +

{
    "count": 1,
    "unitPriceExcludingVatCurrency": product["priceExcludingVatCurrency"],
    "vatType": product["vatType"],
    "sortIndex": i,
}

# +

all_invoices = requests.get(
    f"{BASE_URL}/invoice",
    auth=AUTH,
    params={"invoiceDateFrom": "1970-01-01", "invoiceDateTo": "2027-12-31"},
).json()
all_invoices

# +

for invoice in all_invoices["values"]:
    print(requests.delete(f"{BASE_URL}/invoice/{invoice['id']}", auth=AUTH).json())
    break
