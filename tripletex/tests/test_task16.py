"""End-to-end test for task 16 (time logging + project invoice).

Environment assumptions tested here:
  - Customer, employee, project with activity all exist.

Scoring assumptions tested here:
  - Timesheet entry created for employee on project/activity.
  - Invoice created with amount = hours × hourly_rate.

Not tested:
  - Whether hourly rate must be set on the project via /project/hourlyRates.
  - Whether invoice must be generated via order/:invoice vs direct POST /invoice.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials, get_current_year_month_day_utc
from tripletex.parsers.task16_time_logging_invoice import TimeLoggingInvoice


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task16_solve_hardcoded():
    """Create a project with activity, then log time and invoice."""
    creds = _sandbox_creds()
    import time
    today = get_current_year_month_day_utc()
    suffix = str(int(time.time()))[-6:]
    project_name = f"Task16 Test {suffix}"

    # Setup: create a project and link an existing "Analyse" activity
    # Use admin employee (has PM rights) and Fjordtech AS as customer

    # Find existing Analyse activity (or create with unique name)
    act_r = requests.get(
        f"{creds.base_url}/activity",
        auth=creds.auth,
        params={"fields": "id,name"},
    )
    act_r.raise_for_status()
    analyse_acts = [a for a in act_r.json()["values"] if a["name"] == "Analyse"]
    if analyse_acts:
        activity_id = analyse_acts[0]["id"]
    else:
        new_act = requests.post(
            f"{creds.base_url}/activity",
            auth=creds.auth,
            json={"name": "Analyse", "activityType": "PROJECT_GENERAL_ACTIVITY", "isProjectActivity": True},
        )
        new_act.raise_for_status()
        activity_id = new_act.json()["value"]["id"]

    proj_r = requests.post(
        f"{creds.base_url}/project",
        auth=creds.auth,
        json={
            "name": project_name,
            "projectManager": {"id": 18441790},  # admin
            "customer": {"id": 108148995},  # Fjordtech AS
            "startDate": today,
            "projectActivities": [
                {"activity": {"id": activity_id}}
            ],
        },
    )
    proj_r.raise_for_status()

    parsed = TimeLoggingInvoice(
        hours=10,
        first_name="Me",
        last_name="Admin",
        email="me@dilawar.ai",
        activity_name="Analyse",
        project_name=project_name,
        customer_name="Fjordtech AS",
        org_number="912345678",
        hourly_rate=1200.0,
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: timesheet entry exists ───────────────────────────────────
    ts = requests.get(
        f"{creds.base_url}/timesheet/entry",
        auth=creds.auth,
        params={"dateFrom": today, "dateTo": get_current_year_month_day_utc(days_offset_forward=1), "fields": "id,hours,project(id,name)", "count": 100},
    ).json()["values"]
    match = [e for e in ts if project_name in e["project"].get("name", "")]
    assert len(match) >= 1, f"Timesheet entry not found. Got: {ts}"
    assert match[-1]["hours"] == 10.0

    # ── Verify: invoice exists with correct amount ───────────────────────
    invoices = requests.get(
        f"{creds.base_url}/invoice",
        auth=creds.auth,
        params={
            "invoiceDateFrom": today,
            "invoiceDateTo": get_current_year_month_day_utc(days_offset_forward=1),
            "fields": "id,amountExcludingVat",
        },
    ).json()["values"]
    # 10 hours × 1200 = 12000
    match_inv = [i for i in invoices if i["amountExcludingVat"] == 12000.0]
    assert len(match_inv) >= 1, f"Invoice for 12000 not found. Got: {invoices}"


def test_task16_parse():
    parsed = TimeLoggingInvoice.parse(
        'Log 15 hours for Samuel Williams (samuel.williams@example.org) on the activity "Analyse" in the project "Website Redesign" for Windmill Ltd (org no. 898523942). Hourly rate: 1950 NOK/h. Generate a project invoice to the customer based on the logged hours.'
    )
    assert parsed.hours == 15
    assert parsed.email == "samuel.williams@example.org"
    assert parsed.activity_name == "Analyse"
    assert parsed.project_name == "Website Redesign"
    assert parsed.org_number == "898523942"
    assert parsed.hourly_rate == 1950.0
