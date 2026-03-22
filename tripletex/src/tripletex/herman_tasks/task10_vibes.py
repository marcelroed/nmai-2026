import requests

from tripletex.config import AUTH, BASE_URL
from tripletex.log_analysis._3_find_by_task import get_task

tasks, raw_data = get_task("Task 10", only_log_version_2=True)

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
all_customers["extra"]["data"]["values"]

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
all_products["extra"]["data"]["values"]

# +

(all_orders,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /order")
]
all_orders["extra"]["data"]["values"]

# +

orders = requests.get(
    f"{BASE_URL}/order",
    auth=AUTH,
    params={"orderDateFrom": "1970-01-01", "orderDateTo": "2027-12-31"},
).json()["values"]
orders

# +

for order in orders[-5:]:
    print(order)
    print()

__import__("rich").print(orders)
