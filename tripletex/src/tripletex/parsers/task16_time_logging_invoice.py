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
    'Erfassen Sie 18 Stunden für Sophia Schmidt (sophia.schmidt@example.org) auf der Aktivität "Design" im Projekt "Sicherheitsaudit" für Windkraft GmbH (Org.-Nr. 882984826). Stundensatz: 950 NOK/h. Erstellen Sie eine Projektrechnung an den Kunden basierend auf den erfassten Stunden.',
    'Registe 4 horas para Maria Ferreira (maria.ferreira@example.org) na atividade "Utvikling" do projeto "Desenvolvimento de app" para Estrela Lda (org. nº 909621682). Taxa horária: 1050 NOK/h. Gere uma fatura de projeto ao cliente com base nas horas registadas.',
    'Registe 35 horas para Inês Ferreira (ines.ferreira@example.org) na atividade "Testing" do projeto "Configuração cloud" para Floresta Lda (org. nº 949247589). Taxa horária: 1000 NOK/h. Gere uma fatura de projeto ao cliente com base nas horas registadas.',
    'Log 15 hours for Samuel Williams (samuel.williams@example.org) on the activity "Analyse" in the project "Website Redesign" for Windmill Ltd (org no. 898523942). Hourly rate: 1950 NOK/h. Generate a project invoice to the customer based on the logged hours.',
    'Registe 27 horas para Ana Sousa (ana.sousa@example.org) na atividade "Utvikling" do projeto "Migração de dados" para Montanha Lda (org. nº 870501366). Taxa horária: 800 NOK/h. Gere uma fatura de projeto ao cliente com base nas horas registadas.',
    'Log 34 hours for Charlotte Williams (charlotte.williams@example.org) on the activity "Analyse" in the project "Security Audit" for Windmill Ltd (org no. 851492623). Hourly rate: 1300 NOK/h. Generate a project invoice to the customer based on the logged hours.',
    'Erfassen Sie 18 Stunden für Mia Meyer (mia.meyer@example.org) auf der Aktivität "Testing" im Projekt "Sicherheitsaudit" für Nordlicht GmbH (Org.-Nr. 934651995). Stundensatz: 1400 NOK/h. Erstellen Sie eine Projektrechnung an den Kunden basierend auf den erfassten Stunden.',
    'Enregistrez 12 heures pour Inès Thomas (ines.thomas@example.org) sur l\'activité "Utvikling" du projet "Mise à niveau système" pour Prairie SARL (nº org. 879748429). Taux horaire : 900 NOK/h. Générez une facture de projet au client basée sur les heures enregistrées.',
    'Registrer 7 timar for Liv Brekke (liv.brekke@example.org) på aktiviteten "Design" i prosjektet "Datamigrering" for Strandvik AS (org.nr 962818684). Timesats: 1650 kr/t. Generer ein prosjektfaktura til kunden basert på dei registrerte timane.',
    'Registe 30 horas para Leonor Almeida (leonor.almeida@example.org) na atividade "Utvikling" do projeto "Migração de dados" para Estrela Lda (org. nº 911889307). Taxa horária: 1550 NOK/h. Gere uma fatura de projeto ao cliente com base nas horas registadas.',
]

logger = configure_logging()


class TimeLoggingInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours: int
    first_name: str
    last_name: str
    email: EmailStr
    activity_name: str
    project_name: str
    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    hourly_rate: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=TimeLoggingInvoice,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Customer exists (by org number).
        #   - Employee exists (by email).
        #   - Project exists (by name, linked to customer) with an activity
        #     matching activity_name.
        #
        # Assumptions about what scoring checks:
        #   - Timesheet entry exists for the employee on the project/activity.
        #   - Invoice created for hours × hourly_rate.
        #
        # Not verified:
        #   - Whether the hourly rate needs to be set on the project via
        #     /project/hourlyRates, or if just the invoice amount is enough.
        #   - Whether the invoice must be generated via order/:invoice or
        #     can be a direct POST /invoice with project reference.
        today = get_current_year_month_day_utc()

        # ── GET lookups ──────────────────────────────────────────────────
        customer = get_customer_by_org_number(self.org_number, tripletex_client)
        customer_id = customer["id"]
        logger.info("task16.customer.found", extra={"customer_id": customer_id})

        employee = get_employee(self.email, tripletex_client)
        employee_id = employee["id"]
        logger.info("task16.employee.found", extra={"employee_id": employee_id})

        # Find the project by name
        projects_r = tripletex_client.get(
            "/project",
            params={"fields": "id,name,displayName,projectActivities(id)"},
        )
        projects_r.raise_for_status()
        projects = projects_r.json()["values"]
        (project,) = [p for p in projects if self.project_name in p["displayName"]]
        project_id = project["id"]
        logger.info(
            "task16.project.found",
            extra={"project_id": project_id, "project_name": project["displayName"]},
        )

        # Find the activity on this project
        activity_id = None
        for pa in project["projectActivities"]:
            pa_r = tripletex_client.get(
                f"/project/projectActivity/{pa['id']}",
                params={"fields": "activity(id,name)"},
            )
            pa_r.raise_for_status()
            pa_data = pa_r.json()["value"]
            if pa_data["activity"]["name"] == self.activity_name:
                activity_id = pa_data["activity"]["id"]
                break

        if activity_id is None:
            # Activity not found on project — check global activities
            act_r = tripletex_client.get("/activity", params={"fields": "id,name"})
            act_r.raise_for_status()
            activities = act_r.json()["values"]
            (activity,) = [a for a in activities if a["name"] == self.activity_name]
            activity_id = activity["id"]

        logger.info("task16.activity.found", extra={"activity_id": activity_id})

        # ── WRITE 1: log timesheet entry ─────────────────────────────────
        ts_payload = {
            "employee": {"id": employee_id},
            "project": {"id": project_id},
            "activity": {"id": activity_id},
            "date": today,
            "hours": self.hours,
        }
        logger.info("task16.timesheet.creating", extra={"payload": ts_payload})
        ts_r = tripletex_client.post("/timesheet/entry", json=ts_payload)
        logger.info(
            "task16.timesheet.response",
            extra={"status": ts_r.status_code, "body": ts_r.json()},
        )
        ts_r.raise_for_status()

        # ── WRITE 2: create invoice for hours × rate ─────────────────────
        invoice_amount = self.hours * self.hourly_rate
        invoice_payload = {
            "invoiceDate": today,
            "invoiceDueDate": get_current_year_month_day_utc(days_offset_forward=30),
            "customer": {"id": customer_id},
            "orders": [
                {
                    "customer": {"id": customer_id},
                    "project": {"id": project_id},
                    "orderDate": today,
                    "deliveryDate": today,
                    "orderLines": [
                        {
                            "count": self.hours,
                            "unitPriceExcludingVatCurrency": self.hourly_rate,
                            "vatType": {"id": 3},  # 25% utgående
                            "description": f"{self.activity_name} - {self.project_name}",
                        }
                    ],
                }
            ],
        }
        logger.info(
            "task16.invoice.creating",
            extra={"payload": invoice_payload, "total": invoice_amount},
        )
        inv_r = tripletex_client.post("/invoice", json=invoice_payload)
        logger.info(
            "task16.invoice.response",
            extra={"status": inv_r.status_code, "body": inv_r.json()},
        )
        inv_r.raise_for_status()

        logger.info("task16.completed")


if __name__ == "__main__":
    for prompt in prompts[:1]:
        res = TimeLoggingInvoice.parse(prompt)
        print(f"{prompt[:80]}...")
        print(f"  {res=}")
        print()
