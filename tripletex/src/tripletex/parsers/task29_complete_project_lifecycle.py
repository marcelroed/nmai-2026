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
    "Gjennomfør heile prosjektsyklusen for 'Dataplattform Strandvik' (Strandvik AS, org.nr 982465958): 1) Prosjektet har budsjett 412400 kr. 2) Registrer timar: Olav Stølsvik (prosjektleiar, olav.stlsvik@example.org) 29 timar og Jorunn Eide (konsulent, jorunn.eide@example.org) 100 timar. 3) Registrer leverandørkostnad 49600 kr frå Skogheim AS (org.nr 950515430). 4) Opprett kundefaktura for prosjektet.",
    "Gjennomfør hele prosjektsyklusen for 'ERP-implementering Snøhetta' (Snøhetta AS, org.nr 954447499): 1) Prosjektet har budsjett 431600 kr. 2) Registrer timer: Sigurd Johansen (prosjektleder, sigurd.johansen@example.org) 47 timer og Erik Haugen (konsulent, erik.haugen@example.org) 46 timer. 3) Registrer leverandørkostnad 95050 kr fra Nordhav AS (org.nr 957929974). 4) Opprett kundefaktura for prosjektet.",
    "Execute the complete project lifecycle for 'Data Platform Ridgepoint' (Ridgepoint Ltd, org no. 808812096): 1) The project has a budget of 316550 NOK. 2) Log time: Daniel Harris (project manager, daniel.harris@example.org) 54 hours and Grace Johnson (consultant, grace.johnson@example.org) 78 hours. 3) Register supplier cost of 58200 NOK from Ironbridge Ltd (org no. 814796019). 4) Create a customer invoice for the project.",
    "Ejecute el ciclo de vida completo del proyecto 'Migración Cloud Montaña' (Montaña SL, org. nº 903805021): 1) El proyecto tiene un presupuesto de 219800 NOK. 2) Registre horas: Ana Rodríguez (director de proyecto, ana.rodriguez@example.org) 36 horas y Isabel García (consultor, isabel.garcia@example.org) 69 horas. 3) Registre costo de proveedor de 46050 NOK de Luna SL (org. nº 988327581). 4) Cree una factura al cliente por el proyecto.",
    "Gjennomfør heile prosjektsyklusen for 'Dataplattform Sjøbris' (Sjøbris AS, org.nr 868541946): 1) Prosjektet har budsjett 361050 kr. 2) Registrer timar: Eirik Stølsvik (prosjektleiar, eirik.stlsvik@example.org) 41 timar og Geir Aasen (konsulent, geir.aasen@example.org) 122 timar. 3) Registrer leverandørkostnad 23500 kr frå Elvdal AS (org.nr 964202133). 4) Opprett kundefaktura for prosjektet.",
    "Execute the complete project lifecycle for 'Cloud Migration Northwave' (Northwave Ltd, org no. 932075482): 1) The project has a budget of 396900 NOK. 2) Log time: Samuel Brown (project manager, samuel.brown@example.org) 74 hours and Sarah Lewis (consultant, sarah.lewis@example.org) 85 hours. 3) Register supplier cost of 56750 NOK from Clearwater Ltd (org no. 889264985). 4) Create a customer invoice for the project.",
    "Gjennomfør hele prosjektsyklusen for 'ERP-implementering Bergvik' (Bergvik AS, org.nr 886943407): 1) Prosjektet har budsjett 400950 kr. 2) Registrer timer: Ragnhild Strand (prosjektleder, ragnhild.strand@example.org) 35 timer og Silje Bakken (konsulent, silje.bakken@example.org) 79 timer. 3) Registrer leverandørkostnad 94350 kr fra Brattli AS (org.nr 810297891). 4) Opprett kundefaktura for prosjektet.",
    "Gjennomfør heile prosjektsyklusen for 'Dataplattform Skogheim' (Skogheim AS, org.nr 841795067): 1) Prosjektet har budsjett 258650 kr. 2) Registrer timar: Torbjørn Brekke (prosjektleiar, torbjrn.brekke@example.org) 64 timar og Arne Kvamme (konsulent, arne.kvamme@example.org) 87 timar. 3) Registrer leverandørkostnad 77950 kr frå Nordlys AS (org.nr 894689668). 4) Opprett kundefaktura for prosjektet.",
    "Exécutez le cycle de vie complet du projet 'Portail Numérique Étoile' (Étoile SARL, nº org. 834437961) : 1) Le projet a un budget de 383650 NOK. 2) Enregistrez le temps : Jade Martin (chef de projet, jade.martin@example.org) 53 heures et Louis Robert (consultant, louis.robert@example.org) 56 heures. 3) Enregistrez un coût fournisseur de 90100 NOK de Montagne SARL (nº org. 891743882). 4) Créez une facture client pour le projet.",
    'Exécutez le cycle de vie complet du projet \'Mise à Niveau Système Soleil\' (Soleil SARL, nº org. 869871079) : 1) Le projet a un budget de 475350 NOK. 2) Enregistrez le temps : Gabriel Richard (chef de projet, gabriel.richard@example.org) 25 heures et Chloé Richard (consultant, chloe.richard@example.org) 96 heures. 3) Enregistrez un coût fournisseur de 93950 NOK de Forêt SARL (nº org. 967857920). 4) Créez une facture client pour le projet.',
    'Execute o ciclo de vida completo do projeto \'Implementação ERP Oceano\' (Oceano Lda, org. nº 924696788): 1) O projeto tem um orçamento de 240850 NOK. 2) Registe horas: Bruno Martins (gestor de projeto, bruno.martins@example.org) 27 horas e Tiago Ferreira (consultor, tiago.ferreira@example.org) 89 horas. 3) Registe custo de fornecedor de 53650 NOK de Porto Alegre Lda (org. nº 957407331). 4) Crie uma fatura ao cliente para o projeto.',
    'Ejecute el ciclo de vida completo del proyecto \'Actualización Sistema Costa\' (Costa Brava SL, org. nº 948221934): 1) El proyecto tiene un presupuesto de 433850 NOK. 2) Registre horas: Diego Martínez (director de proyecto, diego.martinez@example.org) 47 horas y Fernando Pérez (consultor, fernando.perez@example.org) 124 horas. 3) Registre costo de proveedor de 98650 NOK de Luna SL (org. nº 851230610). 4) Cree una factura al cliente por el proyecto.',
]

logger = configure_logging()


class TeamMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_name: str
    last_name: str
    email: EmailStr
    hours: int


class CompleteProjectLifecycle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    customer_name: str
    customer_org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    budget: float
    project_manager: TeamMember
    consultant: TeamMember
    supplier_cost: float
    supplier_name: str
    supplier_org_number: Annotated[str, Field(pattern=r"^\d{9}$")]

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CompleteProjectLifecycle,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Customer exists (by org number).
        #   - PM and consultant employees exist (by email).
        #   - Supplier exists (by org number) — but supplier cost is posted as
        #     a voucher, not through supplier invoice, so supplier lookup may not
        #     be needed.
        #   - Standard accounts: 4300 (innkjøp tjenester), 1920 (bank).
        #
        # Assumptions about what scoring checks:
        #   - Project created with correct name, linked to customer, with budget.
        #   - Time entries exist for PM and consultant on the project.
        #   - Supplier cost voucher posted on the project.
        #   - Customer invoice created for the project.
        #
        # Not verified:
        #   - Whether budget is set via project.fixedprice or a separate field.
        #   - Whether the supplier cost account (4300) is correct.
        #   - Whether the invoice needs specific order lines or just project reference.
        #   - Whether timesheet hours need a specific activity.
        today = get_current_year_month_day_utc()

        # ── GET lookups (all free) ───────────────────────────────────────
        customer = get_customer_by_org_number(
            self.customer_org_number, tripletex_client
        )
        customer_id = customer["id"]
        logger.info("task29.customer.found", extra={"customer_id": customer_id})

        pm_employee = get_employee(self.project_manager.email, tripletex_client)
        pm_id = pm_employee["id"]
        logger.info("task29.pm.found", extra={"pm_id": pm_id})

        consultant_employee = get_employee(self.consultant.email, tripletex_client)
        consultant_id = consultant_employee["id"]
        logger.info("task29.consultant.found", extra={"consultant_id": consultant_id})

        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}

        # ── WRITE 1: create project with activity ────────────────────────
        project_payload = {
            "name": self.project_name,
            "projectManager": {"id": pm_id},
            "customer": {"id": customer_id},
            "isInternal": False,
            "startDate": today,
            "fixedprice": self.budget,
            "projectActivities": [
                {
                    "activity": {
                        "name": self.project_name,
                        "activityType": "PROJECT_GENERAL_ACTIVITY",
                        "isProjectActivity": True,
                    }
                }
            ],
        }
        logger.info("task29.project.creating", extra={"payload": project_payload})
        proj_r = tripletex_client.post("/project", json=project_payload)
        logger.info(
            "task29.project.response",
            extra={"status": proj_r.status_code, "body": proj_r.json()},
        )
        proj_r.raise_for_status()
        project = proj_r.json()["value"]
        project_id = project["id"]
        activity_id = project["projectActivities"][0]["id"]

        # Get the actual activity ID from the projectActivity link
        pa_r = tripletex_client.get(
            f"/project/projectActivity/{activity_id}",
            params={"fields": "activity(id)"},
        )
        pa_r.raise_for_status()
        real_activity_id = pa_r.json()["value"]["activity"]["id"]
        logger.info(
            "task29.project.created",
            extra={
                "project_id": project_id,
                "activity_id": real_activity_id,
            },
        )

        # ── WRITE 2: batch log time for PM + consultant ──────────────────
        timesheet_payload = [
            {
                "employee": {"id": pm_id},
                "project": {"id": project_id},
                "activity": {"id": real_activity_id},
                "date": today,
                "hours": self.project_manager.hours,
            },
            {
                "employee": {"id": consultant_id},
                "project": {"id": project_id},
                "activity": {"id": real_activity_id},
                "date": today,
                "hours": self.consultant.hours,
            },
        ]
        logger.info("task29.timesheet.creating", extra={"payload": timesheet_payload})
        ts_r = tripletex_client.post("/timesheet/entry/list", json=timesheet_payload)
        logger.info(
            "task29.timesheet.response",
            extra={"status": ts_r.status_code, "body": ts_r.json()},
        )
        ts_r.raise_for_status()

        # ── WRITE 3: post supplier cost voucher ──────────────────────────
        supplier_voucher = {
            "date": today,
            "description": f"Leverandørkostnad {self.supplier_name}",
            "postings": [
                {
                    "account": {"id": accts[4300]["id"]},
                    "amountCurrency": self.supplier_cost,
                    "amountGross": self.supplier_cost,
                    "amountGrossCurrency": self.supplier_cost,
                    "project": {"id": project_id},
                    "row": 1,
                },
                {
                    "account": {"id": accts[1920]["id"]},
                    "amountCurrency": -self.supplier_cost,
                    "amountGross": -self.supplier_cost,
                    "amountGrossCurrency": -self.supplier_cost,
                    "row": 2,
                },
            ],
        }
        logger.info("task29.supplierCost.creating", extra={"payload": supplier_voucher})
        sc_r = tripletex_client.post("/ledger/voucher", json=supplier_voucher)
        logger.info(
            "task29.supplierCost.response",
            extra={"status": sc_r.status_code, "body": sc_r.json()},
        )
        sc_r.raise_for_status()

        # ── WRITE 4: create customer invoice for the project ─────────────
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
                            "count": 1,
                            "unitPriceExcludingVatCurrency": self.budget,
                            "vatType": {"id": 3},  # 25% utgående
                            "description": self.project_name,
                        }
                    ],
                }
            ],
        }
        logger.info("task29.invoice.creating", extra={"payload": invoice_payload})
        inv_r = tripletex_client.post("/invoice", json=invoice_payload)
        logger.info(
            "task29.invoice.response",
            extra={"status": inv_r.status_code, "body": inv_r.json()},
        )
        inv_r.raise_for_status()

        logger.info("task29.completed")


if __name__ == "__main__":
    for prompt in prompts[:1]:
        res = CompleteProjectLifecycle.parse(prompt)
        print(f"{prompt[:80]}...")
        print(f"  {res=}")
        print()
