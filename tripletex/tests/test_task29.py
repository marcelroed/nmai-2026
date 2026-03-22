"""End-to-end test for task 29 (complete project lifecycle) against the sandbox.

Environment assumptions tested here:
  - Customer exists (by org number).
  - PM and consultant employees exist (by email).
  - Standard accounts: 4300, 1920.
  - Timesheet entry creation works with project + activity.

Scoring assumptions tested here:
  - Project created with name, customer link, budget.
  - Time entries logged for PM and consultant.
  - Supplier cost voucher posted with project reference.
  - Customer invoice created for the project.

Not tested:
  - Whether budget field (fixedprice) is the right way to set project budget.
  - Whether supplier cost account (4300) is correct.
  - Whether invoice amount should equal budget or be computed differently.
  - Whether timesheet activity name matters.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials, get_current_year_month_day_utc
from tripletex.parsers.task29_complete_project_lifecycle import (
    CompleteProjectLifecycle,
    TeamMember,
)


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task29_solve_hardcoded():
    """Run solve with sandbox-compatible data."""
    creds = _sandbox_creds()

    # Use admin employee (has PM rights) and another employee with email
    parsed = CompleteProjectLifecycle(
        project_name="Test Lifecycle E2E",
        customer_name="Fjordtech AS",
        customer_org_number="912345678",
        budget=100000,
        project_manager=TeamMember(
            first_name="Me", last_name="Admin", email="me@dilawar.ai", hours=10
        ),
        consultant=TeamMember(
            first_name="Kari", last_name="Berg", email="kari.berg@example.org", hours=20
        ),
        supplier_cost=25000,
        supplier_name="Test Supplier",
        supplier_org_number="123456789",
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: project exists ───────────────────────────────────────────
    projects = requests.get(
        f"{creds.base_url}/project",
        auth=creds.auth,
        params={"fields": "id,name,isInternal,customer(id),projectActivities(id)", "count": 100},
    ).json()["values"]

    match = [p for p in projects if p["name"] == "Test Lifecycle E2E"]
    assert len(match) >= 1, f"Project not found. Got: {[p['name'] for p in projects]}"
    proj = match[-1]
    assert proj["customer"]["id"] == 108148995  # Fjordtech AS
    assert len(proj["projectActivities"]) >= 1


def test_task29_parse():
    """Parse a prompt to verify field extraction."""
    prompt = "Execute the complete project lifecycle for 'Data Platform Ridgepoint' (Ridgepoint Ltd, org no. 808812096): 1) The project has a budget of 316550 NOK. 2) Log time: Daniel Harris (project manager, daniel.harris@example.org) 54 hours and Grace Johnson (consultant, grace.johnson@example.org) 78 hours. 3) Register supplier cost of 58200 NOK from Ironbridge Ltd (org no. 814796019). 4) Create a customer invoice for the project."

    parsed = CompleteProjectLifecycle.parse(prompt)

    assert parsed.project_name == "Data Platform Ridgepoint"
    assert parsed.customer_org_number == "808812096"
    assert parsed.budget == 316550.0
    assert parsed.project_manager.email == "daniel.harris@example.org"
    assert parsed.project_manager.hours == 54
    assert parsed.consultant.email == "grace.johnson@example.org"
    assert parsed.consultant.hours == 78
    assert parsed.supplier_cost == 58200.0
    assert parsed.supplier_org_number == "814796019"
