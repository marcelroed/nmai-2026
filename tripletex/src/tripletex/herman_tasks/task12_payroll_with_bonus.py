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
self_tripletex_creds = TripletexCredentials.placeholder_TODO()
self_salary_amount = 4321
self_bonus_amount = 1234

# +

employee = get_employee(
    employee_email=self_employee_email, tripletex_creds=self_tripletex_creds
)
employee

# +

salary_type = requests.get(
    f"{self_tripletex_creds.base_url}/salary/type",
    auth=(
        0,
        self_tripletex_creds.session_token,
    ),  # params={"employeeIds":employee['id']}
).json()["values"]
logger.info("got the salary types", extra={"salary_type": salary_type})

# +

(fixed_salary_type,) = [x for x in salary_type if x["name"] == "Fastlønn"]
(bonus_salary_type,) = [x for x in salary_type if x["name"] == "Bonus"]

# +

date = get_current_year_month_day_utc()
year = int(get_current_year_month_day_utc()[:4])
month = int(get_current_year_month_day_utc()[5:7])

json_data = {
    "date": date,
    "year": year,
    "month": month,
    "payslips": [
        {
            "employee": {"id": employee["id"]},
            "date": get_current_year_month_day_utc(),
            "year": year,
            "month": month,
            "specifications": [
                {
                    "salaryType": {"id": fixed_salary_type["id"]},
                    "year": year,
                    "month": month,
                    "count": 1,
                    "rate": self_salary_amount,
                },
                {
                    "salaryType": {"id": bonus_salary_type["id"]},
                    "year": year,
                    "month": month,
                    "count": 1,
                    "rate": self_bonus_amount,
                },
            ],
        }
    ],
}
json_data

# +

salary_transaction = requests.post(
    f"{self_tripletex_creds.base_url}/salary/transaction",
    auth=(0, self_tripletex_creds.session_token),
    json=json_data,
).json()
salary_transaction
