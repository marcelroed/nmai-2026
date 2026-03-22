import requests

from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_current_year_month_day_utc,
    get_employee,
)
from tripletex.my_log import configure_logging

logger = configure_logging()

# +

self_employee_email = "fernando.garcia@example.org"
self_travel_expense_title = "Customer visit Trondheim"
self_tripletex_creds = TripletexCredentials.placeholder_TODO()
self_flight_cost = 4321
self_taxi_cost = 124
self_title = "Visit Tromso"
self_location = "Tromso"
self_total_days = 12
self_per_diem_budget = 800

# +

employee = get_employee(
    employee_email=self_employee_email, tripletex_creds=self_tripletex_creds
)
employee

# +

(payment_type,) = requests.get(
    f"{self_tripletex_creds.base_url}/travelExpense/paymentType",
    auth=self_tripletex_creds.auth,
).json()["values"]

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

(travel_expense_rates_filtered_by_rate,) = [
    x for x in travel_expense_rates if x["rate"] == self_per_diem_budget
]

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

# +

travel_expense = requests.post(
    f"{self_tripletex_creds.base_url}/travelExpense",
    auth=self_tripletex_creds.auth,
    json=json_data,
).json()
travel_expense
