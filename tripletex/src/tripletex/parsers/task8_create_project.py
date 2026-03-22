from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_current_year_month_day_utc,
    get_customer_by_org_number,
    get_employee,
)
from tripletex.my_log import configure_logging

prompts = [
    'Crie o projeto "Análise Porto" vinculado ao cliente Porto Alegre Lda (org. nº 996943305). O gerente de projeto é Lucas Oliveira (lucas.oliveira@example.org).',
    'Opprett prosjektet "Migrasjon Snøhetta" knyttet til kunden Snøhetta AS (org.nr 842796032). Prosjektleder er Eline Berg (eline.berg@example.org).',
    'Opprett prosjektet "Analyse Fjordkraft" knyttet til kunden Fjordkraft AS (org.nr 944845712). Prosjektleder er Hilde Johansen (hilde.johansen@example.org).',
    'Opprett prosjektet "Integrasjon Vestfjord" knytt til kunden Vestfjord AS (org.nr 954285405). Prosjektleiar er Håkon Eide (hakon.eide@example.org).',
    'Erstellen Sie das Projekt "Integration Bergwerk" verknüpft mit dem Kunden Bergwerk GmbH (Org.-Nr. 986555080). Projektleiter ist Leon Meyer (leon.meyer@example.org).',
    'Créez le projet "Analyse Lumière" lié au client Lumière SARL (nº org. 929902688). Le chef de projet est Léo Dubois (leo.dubois@example.org).',
    'Créez le projet "Migration Étoile" lié au client Étoile SARL (nº org. 964531161). Le chef de projet est Arthur Dubois (arthur.dubois@example.org).',
    'Erstellen Sie das Projekt "Integration Windkraft" verknüpft mit dem Kunden Windkraft GmbH (Org.-Nr. 804172807). Projektleiter ist Hannah Weber (hannah.weber@example.org).',
    'Create the project "Analysis Oakwood" linked to the customer Oakwood Ltd (org no. 849612913). The project manager is Lucy Taylor (lucy.taylor@example.org).',
]

logger = configure_logging()


class CreateProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    project_manager_first_name: str
    project_manager_last_name: str
    project_manager_email: EmailStr

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreateProject,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Customer exists (by org number).
        #   - PM employee exists (by email) and has project manager permissions.
        #
        # Assumptions about what scoring checks:
        #   - Project created with correct name, linked to customer.
        #   - Project manager set to the correct employee.
        today = get_current_year_month_day_utc()

        # ── GET lookups ──────────────────────────────────────────────────
        customer = get_customer_by_org_number(self.org_number, tripletex_client)
        logger.info("task8.customer.found", extra={"customer_id": customer["id"]})

        pm = get_employee(self.project_manager_email, tripletex_client)
        logger.info("task8.pm.found", extra={"pm_id": pm["id"]})

        # ── WRITE: create project ────────────────────────────────────────
        project_payload = {
            "name": self.project_name,
            "projectManager": {"id": pm["id"]},
            "customer": {"id": customer["id"]},
            "startDate": today,
        }
        logger.info("task8.project.creating", extra={"payload": project_payload})
        r = tripletex_client.post("/project", json=project_payload)
        logger.info(
            "task8.project.response",
            extra={"status": r.status_code, "body": r.json()},
        )
        r.raise_for_status()

        logger.info(
            "task8.completed",
            extra={"project_id": r.json()["value"]["id"]},
        )


if __name__ == "__main__":
    for prompt in prompts:
        res = CreateProject.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
