import requests
import requests.exceptions
import rich

from tripletex.config import AUTH, BASE_URL
from tripletex.herman_tasks.utils import get_current_year_month_day_utc
from tripletex.log_analysis._3_find_by_task import get_task

tasks, raw_data = get_task("Task 12", only_log_version_2=True)

# +
self_email = "fernando.garcia@example.org"
random_salary_type_id = salary_type[0]["id"]
random_bonus_type_id = salary_type[1]["id"]
self_salary_amount = 41050
self_bonus_amount = 9800

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

(all_employees,) = [
    x for x in log_data if x["message"].endswith("got tripletex data for /employee")
]
(employee,) = [
    employee
    for employee in all_employees["extra"]["data"]["values"]
    if employee["email"] == "eirik.brekke@example.org"
]
employee

# +

employee

# +

all_ledger_accounts = [
    x
    for x in log_data
    if x["message"].endswith("got tripletex data for /ledger/account")
]
all_ledger_accounts

# +

ledger_account = requests.get(f"{BASE_URL}/ledger/account", auth=AUTH).json()["values"]
str(ledger_account)[:500]

# +


# +

employees = requests.get(f"{BASE_URL}/employee", auth=AUTH).json()["values"]
(employee,) = [employee for employee in employees if employee["email"] == self_email]

# +

salary_type = requests.get(
    f"{BASE_URL}/salary/type",
    auth=AUTH,  # params={"employeeIds":employee['id']}
).json()["values"]
len(salary_type)

# +

requests.get(
    f"{BASE_URL}/salary/payslip",
    auth=AUTH,
    params={"yearFrom": 1970, "monthFrom": 1, "employeeId": employee["id"]},
).json()

requests.get(f"{BASE_URL}/salary/payslip/32629964", auth=AUTH).json()

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
                    "salaryType": {"id": random_salary_type_id},
                    "year": year,
                    "month": month,
                    "count": 1,
                    "rate": self_salary_amount,
                },
                {
                    "salaryType": {"id": random_bonus_type_id},
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
    f"{BASE_URL}/salary/transaction", auth=AUTH, json=json_data
).json()
salary_transaction
