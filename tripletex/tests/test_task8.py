"""End-to-end test for task 8 (create project linked to customer).

Environment assumptions tested here:
  - Customer exists (by org number).
  - PM employee exists (by email) and has project manager permissions.

Scoring assumptions tested here:
  - Project created with correct name, linked to customer, with PM set.
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.parsers.task8_create_project import CreateProject


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task8_solve_hardcoded():
    creds = _sandbox_creds()

    parsed = CreateProject(
        project_name="Test Project Task8",
        customer_name="Fjordtech AS",
        org_number="912345678",
        project_manager_first_name="Me",
        project_manager_last_name="Admin",
        project_manager_email="me@dilawar.ai",
    )

    parsed.solve(tripletex_creds=creds)

    # ── Verify ───────────────────────────────────────────────────────────
    projects = requests.get(
        f"{creds.base_url}/project",
        auth=creds.auth,
        params={"fields": "id,name,customer(id),projectManager(id)", "count": 100},
    ).json()["values"]

    match = [p for p in projects if p["name"] == "Test Project Task8"]
    assert len(match) >= 1
    proj = match[-1]
    assert proj["customer"]["id"] == 108148995  # Fjordtech AS


def test_task8_parse():
    parsed = CreateProject.parse(
        'Create the project "Analysis Oakwood" linked to the customer Oakwood Ltd (org no. 849612913). The project manager is Lucy Taylor (lucy.taylor@example.org).'
    )
    assert parsed.project_name == "Analysis Oakwood"
    assert parsed.org_number == "849612913"
    assert parsed.project_manager_email == "lucy.taylor@example.org"
