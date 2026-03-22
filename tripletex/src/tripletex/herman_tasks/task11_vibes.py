import requests

from tripletex.config import AUTH, BASE_URL
from tripletex.log_analysis._3_find_by_task import get_task

tasks, raw_data = get_task("Task 11", only_log_version_2=True)

# +

task = tasks[0]
log_data = raw_data[task.request_id][::-1]

# +

task.prompt

# +

[(i, x["message"]) for i, x in enumerate(log_data)]

# +

# TODO: log the suppliers
(all_customers,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /customer")
]
all_customers["extra"]["data"]["values"]

# +

vat_types = requests.get(f"{BASE_URL}/ledger/vatType", auth=AUTH).json()
vat_types["values"]

# +

(nok_currency,) = [
    x
    for x in requests.get(f"{BASE_URL}/currency", auth=AUTH).json()["values"]
    if x["displayName"] == "NOK"
]
nok_currency

# +


(all_vats,) = [
    x
    for x in log_data
    if x["message"].endswith("got tripletex data for /ledger/vatType")
]
all_vats["extra"]["data"]["values"]
