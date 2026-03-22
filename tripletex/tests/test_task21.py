"""End-to-end test for task 21 (onboarding from offer PDF) against the sandbox.

Environment assumptions tested here:
  - Department "Salg" pre-exists (solve looks it up, never creates).
  - Exactly one division exists.

Scoring assumptions tested here:
  - Employee created with correct firstName, lastName, dateOfBirth, department.
  - Employment created with correct startDate, percentageOfFullTimeEquivalent, annualSalary.
  - Employment details: employmentType=ORDINARY, employmentForm=PERMANENT.
  - StandardTime created with correct hoursPerDay.

Not tested (cannot test against persistent sandbox):
  - Email format — generated as firstname.lastname@example.org, may not match scorer.
  - Whether scorer checks remunerationType or workingHoursScheme.
"""

import time

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.parsers.task21_onboarding_from_offer_pdf import OnboardingFromOfferPDF


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def _unique_suffix() -> str:
    return str(int(time.time()))[-6:]


def test_task21_solve_hardcoded():
    """Run solve with a hardcoded parsed model (no LLM call) against the sandbox."""
    creds = _sandbox_creds()
    suffix = _unique_suffix()

    parsed = OnboardingFromOfferPDF(
        first_name=f"Testperson{suffix}",
        last_name="Ødegård",
        birth_year=1987,
        birth_month=12,
        birth_day=30,
        job_title="IT-konsulent",
        department="Salg",  # must pre-exist in sandbox
        employment_percentage=100.0,
        annual_salary=560000,
        standard_work_hours_per_day=7.5,
        start_year=2026,
        start_month=5,
        start_day=24,
    )

    parsed.solve(tripletex_client=creds.to_client())

    # ── Verify: employee exists ──────────────────────────────────────────
    employees = requests.get(
        f"{creds.base_url}/employee",
        auth=creds.auth,
        params={"fields": "id,firstName,lastName,dateOfBirth,department(id,name)"},
    ).json()["values"]

    match = [
        e
        for e in employees
        if e["firstName"] == f"Testperson{suffix}" and e["lastName"] == "Ødegård"
    ]
    assert len(match) >= 1, f"Employee not found. Got: {employees}"
    emp = match[-1]
    assert emp["dateOfBirth"] == "1987-12-30"
    assert emp["department"]["name"] == "Salg"

    # ── Verify: employment details ───────────────────────────────────────
    emp_id = emp["id"]
    employments = requests.get(
        f"{creds.base_url}/employee/employment",
        auth=creds.auth,
        params={"employeeId": emp_id, "fields": "id,startDate,employmentDetails(*)"},
    ).json()["values"]

    our_employments = [e for e in employments if e["startDate"] == "2026-05-24"]
    assert len(our_employments) >= 1, f"Employment not found. Got: {employments}"
    employment = our_employments[-1]

    details = requests.get(
        f"{creds.base_url}/employee/employment/details",
        auth=creds.auth,
        params={"fields": "*"},
    ).json()["values"]
    our_details = [d for d in details if d["employment"]["id"] == employment["id"]]
    assert len(our_details) >= 1, f"Employment details not found. Got: {details}"
    detail = our_details[-1]
    assert detail["percentageOfFullTimeEquivalent"] == 100.0
    assert detail["annualSalary"] == 560000.0
    assert detail["employmentType"] == "ORDINARY"
    assert detail["employmentForm"] == "PERMANENT"

    # ── Verify: standard time ────────────────────────────────────────────
    standard_times = requests.get(
        f"{creds.base_url}/employee/standardTime",
        auth=creds.auth,
        params={"employeeId": emp_id, "fields": "*"},
    ).json()["values"]
    assert len(standard_times) >= 1, f"Standard time not found. Got: {standard_times}"
    st = standard_times[-1]
    assert st["hoursPerDay"] == 7.5


def test_task21_parse_and_solve():
    """Parse a real offer letter with the LLM, then solve against the sandbox."""
    from pathlib import Path

    creds = _sandbox_creds()
    attachment = Path("data/files/parsed/tilbudsbrev_nb_05.txt").read_text()
    prompt = "Du har mottatt et tilbudsbrev (se vedlagt PDF) for en ny ansatt. Utfor komplett onboarding: opprett den ansatte, tilknytt riktig avdeling, sett opp ansettelsesforhold med stillingsprosent og arslonn, og konfigurer standard arbeidstid."

    parsed = OnboardingFromOfferPDF.parse(prompt, attachment)

    # Sanity-check parsing against known content of tilbudsbrev_nb_05
    assert parsed.first_name == "Kristian"
    assert parsed.last_name == "Haugen"
    assert parsed.department == "HR"
    assert parsed.employment_percentage == 80.0
    assert parsed.annual_salary == 850000
    assert parsed.standard_work_hours_per_day == 6.0

    parsed.solve(tripletex_client=creds.to_client())
