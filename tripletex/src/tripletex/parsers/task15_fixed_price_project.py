from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_current_year_month_day_utc,
    get_customer_by_org_number,
    get_employee,
)

prompts = [
    'Fixez un prix forfaitaire de 128500 NOK sur le projet "Mise à niveau infrastructure" pour Forêt SARL (nº org. 846715363). Le chef de projet est Camille Martin (camille.martin@example.org). Facturez au client 33 % du prix fixe comme paiement d\'étape.',
    'Set a fixed price of 170500 NOK on the project "Infrastructure Upgrade" for Brightstone Ltd (org no. 850116091). The project manager is Charlotte Walker (charlotte.walker@example.org). Invoice the customer for 33% of the fixed price as a milestone payment.',
    'Legen Sie einen Festpreis von 473250 NOK für das Projekt "Datensicherheit" für Windkraft GmbH (Org.-Nr. 886395582) fest. Projektleiter ist Maximilian Wagner (maximilian.wagner@example.org). Stellen Sie dem Kunden 25 % des Festpreises als Meilensteinzahlung in Rechnung.',
    'Fixez un prix forfaitaire de 328650 NOK sur le projet "Développement e-commerce" pour Cascade SARL (nº org. 913754689). Le chef de projet est Alice Moreau (alice.moreau@example.org). Facturez au client 25 % du prix fixe comme paiement d\'étape.',
    'Sett fastpris 181650 kr på prosjektet "Nettbutikk-utvikling" for Tindra AS (org.nr 870827946). Prosjektleder er Kristian Nilsen (kristian.nilsen@example.org). Fakturer kunden for 50 % av fastprisen som en delbetaling.',
    'Legen Sie einen Festpreis von 350650 NOK für das Projekt "ERP-Implementierung" für Sonnental GmbH (Org.-Nr. 877407047) fest. Projektleiter ist Finn Müller (finn.muller@example.org). Stellen Sie dem Kunden 25 % des Festpreises als Meilensteinzahlung in Rechnung.',
    'Establezca un precio fijo de 266550 NOK en el proyecto "Seguridad de datos" para Montaña SL (org. nº 865036981). El director del proyecto es Sofía Pérez (sofia.perez@example.org). Facture al cliente el 50 % del precio fijo como pago parcial.',
    'Sett fastpris 318800 kr på prosjektet "Digital transformasjon" for Strandvik AS (org.nr 883822684). Prosjektleiar er Jorunn Brekke (jorunn.brekke@example.org). Fakturer kunden for 50 % av fastprisen som ei delbetaling.',
    'Sett fastpris 363850 kr på prosjektet "Nettbutikk-utvikling" for Havbris AS (org.nr 876325497). Prosjektleder er Ingrid Moe (ingrid.moe@example.org). Fakturer kunden for 75 % av fastprisen som en delbetaling.',
    'Sett fastpris 478900 kr på prosjektet "Automatiseringsprosjekt" for Strandvik AS (org.nr 858540402). Prosjektleiar er Bjørn Aasen (bjrn.aasen@example.org). Fakturer kunden for 25 % av fastprisen som ei delbetaling.',
    'Defina um preço fixo de 324900 NOK no projeto "Migração para nuvem" para Cascata Lda (org. nº 904335843). O gestor de projeto é Tiago Santos (tiago.santos@example.org). Fature ao cliente 50 % do preço fixo como pagamento por etapa.',
    'Establezca un precio fijo de 261700 NOK en el proyecto "Integración CRM" para Luna SL (org. nº 847950021). El director del proyecto es Andrés López (andres.lopez@example.org). Facture al cliente el 75 % del precio fijo como pago parcial.',
]


class FixedPriceProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_price: float
    project_name: str
    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    project_manager_first_name: str
    project_manager_last_name: str
    project_manager_email: EmailStr
    invoice_percentage: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=FixedPriceProject,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        customer = get_customer_by_org_number(
            org_number=self.org_number, tripletex_client=tripletex_client
        )

        project_manager_employee = get_employee(
            self.project_manager_email, tripletex_client=tripletex_client
        )

        projects = tripletex_client.get("/project").json()["values"]

        (project,) = [
            project
            for project in projects
            if self.project_name in project["displayName"]
        ]

        project_payload = {
            "version": 0,
            "projectManager": {"id": project_manager_employee["id"]},
            "isFixedPrice": True,
            "fixedprice": self.fixed_price,
        }
        project_payload

        update_project_r = tripletex_client.put(
            f"/project/{project['id']}",
            json=project_payload,
        ).json()
        update_project_r["value"]

        order_payload = {
            "customer": {"id": customer["id"]},
            "project": {"id": project["id"]},
            "orderDate": get_current_year_month_day_utc(),
            "deliveryDate": get_current_year_month_day_utc(),
        }
        order_payload

        order_r = tripletex_client.post("/order", json=order_payload).json()
        order_r["value"]["id"]

        amount_on_account = (
            Decimal(self.fixed_price)
            * Decimal(self.invoice_percentage)
            / Decimal("100")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        invoice_params = {
            "invoiceDate": get_current_year_month_day_utc(),
            "createOnAccount": "WITHOUT_VAT",
            "amountOnAccount": str(amount_on_account),
            "onAccountComment": "Milestone payment",
        }
        invoice_params

        invoice_r = tripletex_client.put(
            f"/order/{order_r['value']['id']}/:invoice",
            params=invoice_params,
        ).json()
        invoice_r


if __name__ == "__main__":
    pass
    # for prompt in prompts:
    #     res = FixedPriceProject.parse(prompt)
    #     print(f"{prompt=}")
    #     print(f"{res=}")
    #     print()
    #     break
