from datetime import date

import requests

from tripletex.herman_tasks.utils import (
    TripletexCredentials,
    get_current_year_month_day_utc,
    get_customer_by_org_number,
    get_employee,
    get_invoice_by_amount_excluding_vat,
)
from tripletex.my_log import configure_logging

logger = configure_logging()

# +

self_tripletex_creds = TripletexCredentials.placeholder_TODO()

self_first_name: str = "William"
self_last_name: str = "Clark"
self_birth_year: int = 1986
self_birth_month: int = 7
self_birth_day: int = 8
self_national_identity_number: str = "08078645254"
self_email: str = "william.clark@example.org"
self_department: str = "Regnskap"
self_occupation_code: str = "1227235"
self_employment_percentage: float = 100.0
self_annual_salary: float = 510000.0
self_start_year: int = 2026
self_start_month: int = 11
self_start_day: int = 30
self_bank_account = "55226841732"

# +

departments = requests.get(
    f"{self_tripletex_creds.base_url}/department", auth=self_tripletex_creds.auth
).json()

# +

department_create_r = requests.post(
    f"{self_tripletex_creds.base_url}/department",
    auth=self_tripletex_creds.auth,
    json={"name": self_department},
).json()
department_create_r

# +

employee_payload = {
    "firstName": self_first_name,
    "lastName": self_last_name,
    "dateOfBirth": f"{self_birth_year}-{self_birth_month:02}-{self_birth_day:02}",
    "email": self_email,
    "nationalIdentityNumber": self_national_identity_number,
    "bankAccountNumber": self_bank_account,
    "department": {"id": department_create_r["value"]["id"]},
    "userType": "STANDARD",
}
employee_payload

# +

employee_post_r = requests.post(
    f"{self_tripletex_creds.base_url}/employee",
    auth=self_tripletex_creds.auth,
    json=employee_payload,
).json()
employee_post_r

# +

employee_employment_payload = {
    "employee": {"id": employee_post_r["value"]["id"]},
    "division": {"id": department_create_r["value"]["id"]},
    "startDate": f"{self_start_year}-{self_start_month:02}-{self_start_day:02}",
    "isMainEmployer": True,
    "employmentDetails": [
        {
            "date": f"{self_start_year}-{self_start_month:02}-{self_start_day:02}",
            "employmentType": "ORDINARY",
            "employmentForm": "PERMANENT",
            "remunerationType": "MONTHLY_WAGE",
            "occupationCode": {"id": self_occupation_code},
            "percentageOfFullTimeEquivalent": self_employment_percentage,
            "annualSalary": self_annual_salary,
        }
    ],
}
employee_employment_payload

# +

employee_employment_r = requests.post(
    f"{self_tripletex_creds.base_url}/employee/employment",
    auth=self_tripletex_creds.auth,
    json=employee_employment_payload,
)
employee_employment_r.json()
