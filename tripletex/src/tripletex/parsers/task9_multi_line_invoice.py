from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_current_year_month_day_utc,
    get_customer_by_org_number,
    get_products_by_product_numbers,
)
from tripletex.my_log import configure_logging

prompts = [
    "Opprett ein faktura til kunden Strandvik AS (org.nr 900314183) med tre produktlinjer: Webdesign (9716) til 13450 kr med 25 % MVA, Skylagring (6906) til 7700 kr med 15 % MVA (næringsmiddel), og Systemutvikling (2265) til 18800 kr med 0 % MVA (avgiftsfri).",
    "Crie uma fatura para o cliente Montanha Lda (org. nº 869972401) com três linhas de produto: Sessão de formação (7733) a 22950 NOK com 25 % IVA, Licença de software (6106) a 10250 NOK com 15 % IVA (alimentos), e Manutenção (1351) a 3150 NOK com 0 % IVA (isento).",
    "Créez une facture pour le client Océan SARL (nº org. 974909103) avec trois lignes de produit : Développement système (9068) à 11000 NOK avec 25 % TVA, Licence logicielle (3111) à 7350 NOK avec 15 % TVA (alimentaire), et Session de formation (9564) à 13150 NOK avec 0 % TVA (exonéré).",
    "Erstellen Sie eine Rechnung für den Kunden Nordlicht GmbH (Org.-Nr. 855854171) mit drei Produktzeilen: Netzwerkdienst (2450) zu 28650 NOK mit 25 % MwSt., Cloud-Speicher (6871) zu 13750 NOK mit 15 % MwSt. (Lebensmittel), und Wartung (2881) zu 18000 NOK mit 0 % MwSt. (befreit).",
    "Crea una factura para el cliente Sierra SL (org. nº 861379760) con tres líneas de producto: Mantenimiento (2109) a 27500 NOK con 25 % IVA, Horas de consultoría (1175) a 3900 NOK con 15 % IVA (alimentos), y Informe de análisis (9974) a 3400 NOK con 0 % IVA (exento).",
    "Crea una factura para el cliente Río Verde SL (org. nº 863477905) con tres líneas de producto: Desarrollo de sistemas (2376) a 12000 NOK con 25 % IVA, Asesoría de datos (1496) a 13450 NOK con 15 % IVA (alimentos), y Mantenimiento (4543) a 12050 NOK con 0 % IVA (exento).",
    "Crie uma fatura para o cliente Solmar Lda (org. nº 857302435) com três linhas de produto: Design web (4982) a 21250 NOK com 25 % IVA, Relatório de análise (8365) a 7100 NOK com 15 % IVA (alimentos), e Sessão de formação (1064) a 9550 NOK com 0 % IVA (isento).",
    'Crie uma fatura para o cliente Floresta Lda (org. nº 944182802) com três linhas de produto: Relatório de análise (2039) a 19600 NOK com 25 % IVA, Design web (3304) a 12450 NOK com 15 % IVA (alimentos), e Licença de software (1599) a 9150 NOK com 0 % IVA (isento).',
]

logger = configure_logging()


class ProductLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    product_number: Annotated[str, Field(pattern=r"^\d+$")]
    amount_excl_vat: float
    vat_rate: float


class MultiLineInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    product_lines: list[ProductLine]

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=MultiLineInvoice,
        )
        assert response.parsed_output
        return response.parsed_output  # ty:ignore[invalid-return-type]

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Customer exists (by org number).
        #   - All 3 products exist (by product number) with correct prices and VAT types.
        #
        # Assumptions about what scoring checks:
        #   - Invoice created with 3 order lines matching the products.
        #   - Each line has correct price and VAT type (from product data).
        logger.info("task9.start")
        logger.info(f"task9.customer.looking_up org={self.org_number}")
        customer = get_customer_by_org_number(
            org_number=self.org_number, tripletex_client=tripletex_client
        )
        logger.info("got customer", extra={"customer": customer})

        products = get_products_by_product_numbers(
            product_numbers=[product.product_number for product in self.product_lines],
            tripletex_client=tripletex_client,
        )
        logger.info("got products", extra={"products": products})

        json_payload = {
            "invoiceDate": get_current_year_month_day_utc(),
            "invoiceDueDate": get_current_year_month_day_utc(days_offset_forward=30),
            "customer": {"id": customer["id"]},
            "orders": [
                {
                    "customer": {"id": customer["id"]},
                    "orderDate": get_current_year_month_day_utc(),
                    "orderLineSorting": "CUSTOM",
                    "deliveryDate": get_current_year_month_day_utc(),
                    "orderLines": [
                        {
                            "count": 1,
                            "unitPriceExcludingVatCurrency": product[
                                "priceExcludingVatCurrency"
                            ],
                            "vatType": product["vatType"],
                            "sortIndex": i,
                            "description": product["displayName"],
                        }
                        for i, product in enumerate(products)
                    ],
                }
            ],
        }
        logger.info("using payload", extra={"payload": json_payload})

        r = tripletex_client.post("/invoice", json=json_payload)
        logger.info(
            "task9.invoice.response",
            extra={"status": r.status_code, "body": r.json()},
        )
        r.raise_for_status()
        logger.info("task9.completed", extra={"invoice_id": r.json()["value"]["id"]})


if __name__ == "__main__":
    from tripletex.herman_tasks.utils import TripletexCredentials

    # for prompt in prompts:
    #     res = MultiLineInvoice.parse(prompt)
    #     print(f"{prompt=}")
    #     print(f"{res=}")
    #     print()
    #     break
    pass
    self = MultiLineInvoice(
        customer_name="Strandvik AS",
        org_number="999000111",
        product_lines=[
            # "4646", "8115-2"
            ProductLine(
                product_name="Webdesign",
                product_number="4646",
                amount_excl_vat=13450.0,
                vat_rate=25.0,
            ),
            ProductLine(
                product_name="Skylagring",
                product_number="8115",
                amount_excl_vat=7700.0,
                vat_rate=15.0,
            ),
        ],
    )
    print(self)
    self.solve(tripletex_client=TripletexCredentials.placeholder_TODO().to_client())
