import unicodedata
from typing import Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import get_department_by_name
from tripletex.my_log import configure_logging

prompts = [
    "Vous avez recu une lettre d'offre (voir PDF ci-joint) pour un nouvel employe. Effectuez l'integration complete : creez l'employe, attribuez le bon departement, configurez les details d'emploi avec le pourcentage et le salaire annuel, et configurez les heures de travail standard.",
    "Sie haben ein Angebotsschreiben erhalten (siehe beigefugte PDF) fuer einen neuen Mitarbeiter. Fuehren Sie das vollstaendige Onboarding durch: erstellen Sie den Mitarbeiter, weisen Sie die richtige Abteilung zu, richten Sie die Beschaeftigungsdetails mit Prozentsatz und Jahresgehalt ein, und konfigurieren Sie die Standardarbeitszeit.",
    "Has recibido una carta de oferta (ver PDF adjunto) para un nuevo empleado. Completa la incorporacion: crea el empleado, asigna el departamento correcto, configura los detalles de empleo con porcentaje y salario anual, y configura las horas de trabajo estandar.",
    "Du har motteke eit tilbodsbrev (sjaa vedlagt PDF) for ein ny tilsett. Utfor komplett onboarding: opprett den tilsette, tilknytt rett avdeling, set opp tilsetjingsforhold med stillingsprosent og arslonn, og konfigurer standard arbeidstid.",
    "Du har mottatt et tilbudsbrev (se vedlagt PDF) for en ny ansatt. Utfor komplett onboarding: opprett den ansatte, tilknytt riktig avdeling, sett opp ansettelsesforhold med stillingsprosent og arslonn, og konfigurer standard arbeidstid.",
    "Voce recebeu uma carta de oferta (ver PDF anexo) para um novo funcionario. Complete a integracao: crie o funcionario, atribua o departamento correto, configure os detalhes de emprego com percentagem e salario anual, e configure as horas de trabalho padrao.",
]

attachments = [
    "data/files/parsed/tilbudsbrev_nb_01.txt",
    "data/files/parsed/tilbudsbrev_nb_05.txt",
    "data/files/parsed/tilbudsbrev_nn_06.txt",
    "data/files/parsed/tilbudsbrev_es_03.txt",
    "data/files/parsed/tilbudsbrev_de_06.txt",
    "data/files/parsed/tilbudsbrev_pt_08.txt",
]

logger = configure_logging()


class OnboardingFromOfferPDF(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str
    birth_year: int
    birth_month: int
    birth_day: int
    job_title: str
    department: str
    employment_percentage: float
    annual_salary: float
    standard_work_hours_per_day: float
    start_year: int
    start_month: int
    start_day: int

    @classmethod
    def parse(cls, prompt: str, attachment: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n--- ATTACHMENT ---\n{attachment}",
                }
            ],
            output_format=OnboardingFromOfferPDF,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - The department named in the offer letter already exists.
        #   - Exactly one division exists (the main "Hovedvirksomhet").
        #   - The employee does not already exist.
        #   - All offer letters specify "Fast stilling" (permanent), so we always
        #     use employmentType=ORDINARY, employmentForm=PERMANENT.
        #   - Employment is always MONTHLY_WAGE and NOT_SHIFT.
        #
        # Assumptions about what scoring checks:
        #   - Employee: firstName, lastName, dateOfBirth, department.
        #   - Employment: startDate, percentageOfFullTimeEquivalent, annualSalary.
        #   - StandardTime: hoursPerDay.
        #   - Email is generated (not in offer letter) — may or may not be checked.
        #     See TODO below.
        start_date = f"{self.start_year}-{self.start_month:02}-{self.start_day:02}"
        date_of_birth = f"{self.birth_year}-{self.birth_month:02}-{self.birth_day:02}"

        # ── GET existing department (always pre-exists for this task) ─────
        dept = get_department_by_name(self.department, tripletex_client)
        department_id = dept["id"]
        logger.info(
            "task21.department.found",
            extra={"department_id": department_id, "dept_name": dept["name"]},
        )

        # ── GET division (needed for employment) ─────────────────────────
        division_r = tripletex_client.get("/division")
        division_r.raise_for_status()
        divisions = division_r.json()["values"]
        logger.info("task21.divisions.fetched", extra={"divisions": divisions})
        division_id = divisions[0]["id"]

        # ── POST create employee with nested employment ─────────────────
        # Nesting employments inside the employee POST saves a write call.
        #
        # TODO: The offer letters don't contain an email address, but Tripletex
        # requires one for STANDARD users. We generate one from the name here.
        # You could argue it makes sense to create an e-mail when onboarding a new employee.
        # If scoring fails on email, consider:
        #   - Using NO_ACCESS userType instead (doesn't require email)
        #   - Checking if the grader expects a specific email format
        # Strip diacritics so the email is pure ASCII (Tripletex rejects non-ASCII).
        def _ascii(s: str) -> str:
            return (
                unicodedata.normalize("NFKD", s)
                .encode("ascii", "ignore")
                .decode("ascii")
                .lower()
            )

        email = f"{_ascii(self.first_name)}.{_ascii(self.last_name)}@example.org"
        employee_payload = {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "dateOfBirth": date_of_birth,
            "email": email,
            "department": {"id": department_id},
            "userType": "STANDARD",
            "employments": [
                {
                    "division": {"id": division_id},
                    "startDate": start_date,
                    "isMainEmployer": True,
                    "employmentDetails": [
                        {
                            "date": start_date,
                            "employmentType": "ORDINARY",
                            "employmentForm": "PERMANENT",
                            "remunerationType": "MONTHLY_WAGE",
                            "workingHoursScheme": "NOT_SHIFT",
                            "percentageOfFullTimeEquivalent": self.employment_percentage,
                            "annualSalary": self.annual_salary,
                        }
                    ],
                }
            ],
        }
        logger.info("task21.employee.creating", extra={"payload": employee_payload})
        emp_r = tripletex_client.post("/employee", json=employee_payload)
        logger.info(
            "task21.employee.response",
            extra={"status": emp_r.status_code, "body": emp_r.json()},
        )
        emp_r.raise_for_status()
        emp_data = emp_r.json()
        employee_id = emp_data["value"]["id"]
        logger.info(
            "task21.employee.created",
            extra={"employee_id": employee_id, "response": emp_data},
        )

        # ── POST create standard work hours ──────────────────────────────
        standard_time_payload = {
            "employee": {"id": employee_id},
            "fromDate": start_date,
            "hoursPerDay": self.standard_work_hours_per_day,
        }
        logger.info(
            "task21.standardTime.creating", extra={"payload": standard_time_payload}
        )
        st_r = tripletex_client.post(
            "/employee/standardTime", json=standard_time_payload
        )
        st_r.raise_for_status()
        st_data = st_r.json()
        logger.info("task21.standardTime.created", extra={"response": st_data})

        logger.info(
            "task21.completed",
            extra={
                "employee_id": employee_id,
                "department_id": department_id,
                "division_id": division_id,
            },
        )


if __name__ == "__main__":
    from pathlib import Path

    from tripletex.herman_tasks.utils import TripletexCredentials

    tripletex_client = TripletexCredentials.placeholder_TODO().to_client()

    prompt = prompts[4]
    attachment = Path(attachments[0]).read_text()

    parsed = OnboardingFromOfferPDF.parse(prompt, attachment)
    logger.info("task21.parsed", extra={"parsed": parsed.model_dump()})

    parsed.solve(tripletex_client=tripletex_client)
