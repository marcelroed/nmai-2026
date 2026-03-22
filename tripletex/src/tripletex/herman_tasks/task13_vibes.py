import requests
import requests.exceptions
import rich

from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_current_year_month_day_utc,
    get_employee,
)
from tripletex.log_analysis._3_find_by_task import get_task

tasks, raw_data = get_task("Task 13", only_log_version_2=True)

# +
self_tripletex_creds = TripletexCredentials.placeholder_TODO()
self_email = "fernando.garcia@example.org"
self_per_diem_budget = 736
self_title = "Kundebesøk Bergen"
self_total_days = 4
self_flight_cost = 1234
self_taxi_cost = 123
self_location = "Tromsø"

task = tasks[0]
log_data = raw_data[task.request_id][::-1]

# +

task.prompt

# +

[(i, x["message"]) for i, x in enumerate(log_data)]

# +

[(i, x["message"]) for i, x in enumerate(log_data) if "ledger" in x["message"]]

# +

# TODO: log the suppliers
(all_customers,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /customer")
]
all_customers["extra"]["data"]["values"]

# +

# (all_employees,) = [
#     x for x in log_data if x["message"].endswith("got tripletex data for /employee")
# ]
# (employee,) = [
#     employee
#     for employee in all_employees["extra"]["data"]["values"]
#     if employee["email"] == self_email
# ]
# employee
employee = get_employee(self_email, tripletex_creds=self_tripletex_creds)

# +

requests.get(
    f"{self_tripletex_creds.base_url}/travelExpense/paymentType",
    auth=self_tripletex_creds.auth,
).json()

# +

travel_expense_categories = requests.get(
    f"{self_tripletex_creds.base_url}/travelExpense/costCategory",
    auth=self_tripletex_creds.auth,
).json()["values"]
(flight_travel_expense_category,) = [
    x for x in travel_expense_categories if x["displayName"] == "Fly"
]
(taxi_expense_category,) = [
    x for x in travel_expense_categories if x["displayName"] == "Taxi"
]

# +

travel_expense_rates = requests.get(
    f"{self_tripletex_creds.base_url}/travelExpense/rate",
    auth=self_tripletex_creds.auth,
    params={"type": "PER_DIEM", "isValidDomestic": True},
).json()["values"]
travel_expense_rates[-1]["id"]  # TODO: actually get the correct per diem rate

# +

travel_expense_rates_filtered_by_rate = [
    x for x in travel_expense_rates if x["rate"] == self_per_diem_budget
][0]
travel_expense_rates_filtered_by_rate

# +

(payment_type,) = requests.get(
    f"{self_tripletex_creds.base_url}/travelExpense/paymentType",
    auth=self_tripletex_creds.auth,
).json()["values"]
payment_type

# +

json_data = {
    "employee": {"id": employee["id"]},
    "title": self_title,
    "travelDetails": {
        "isForeignTravel": False,
        "isDayTrip": False,
        "departureDate": get_current_year_month_day_utc(
            days_offset_forward=-self_total_days
        ),
        "returnDate": get_current_year_month_day_utc(),
        "purpose": self_title,
    },
    "isChargeable": False,
    "isFixedInvoicedAmount": False,
    "isIncludeAttachedReceiptsWhenReinvoicing": False,
    "perDiemCompensations": [
        {
            "rateType": {"id": travel_expense_rates_filtered_by_rate["id"]},
            "count": self_total_days,
            "location": self_location,
        }
    ],
    "costs": [
        {
            "paymentType": {"id": payment_type["id"]},
            "date": get_current_year_month_day_utc(
                days_offset_forward=-self_total_days
            ),
            "costCategory": {"id": flight_travel_expense_category["id"]},
            "amountCurrencyIncVat": self_flight_cost,
            "comments": "flight ticket",
        },
        {
            "paymentType": {"id": payment_type["id"]},
            "date": get_current_year_month_day_utc(
                days_offset_forward=-self_total_days
            ),
            "costCategory": {"id": taxi_expense_category["id"]},
            "amountCurrencyIncVat": self_taxi_cost,
            "comments": "taxi",
        },
    ],
}
rich.print(json_data)

# +

travel_expense = requests.post(
    f"{self_tripletex_creds.base_url}/travelExpense",
    auth=self_tripletex_creds.auth,
    json=json_data,
).json()
travel_expense
