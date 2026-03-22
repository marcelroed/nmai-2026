from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_invoice_by_amount_excluding_vat,
)

prompts = [
    'Kunden Vestfjord AS (org.nr 860678403) har reklamert på fakturaen for "Skylagring" (45350 kr ekskl. MVA). Opprett ei fullstendig kreditnota som reverserer heile fakturaen.',
    'O cliente Cascata Lda (org. nº 967090743) reclamou sobre a fatura referente a "Sessão de formação" (23800 NOK sem IVA). Emita uma nota de crédito completa que reverta toda a fatura.',
    'El cliente Viento SL (org. nº 997137310) ha reclamado sobre la factura por "Desarrollo de sistemas" (47700 NOK sin IVA). Emita una nota de crédito completa que revierta toda la factura.',
    'Le client Montagne SARL (nº org. 882988155) a réclamé concernant la facture pour "Heures de conseil" (40900 NOK HT). Émettez un avoir complet qui annule l\'intégralité de la facture.',
    'El cliente Luna SL (org. nº 993794775) ha reclamado sobre la factura por "Diseño web" (15150 NOK sin IVA). Emita una nota de crédito completa que revierta toda la factura.',
    'El cliente Viento SL (org. nº 857019199) ha reclamado sobre la factura por "Informe de análisis" (27200 NOK sin IVA). Emita una nota de crédito completa que revierta toda la factura.',
    'Kunden Nordlys AS (org.nr 902392165) har reklamert på fakturaen for "Programvarelisens" (47350 kr ekskl. MVA). Opprett ei fullstendig kreditnota som reverserer heile fakturaen.',
]


class CreditNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    invoice_description: str
    amount_excl_vat: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreditNote,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        invoice = get_invoice_by_amount_excluding_vat(
            amount_excluding_vat=int(self.amount_excl_vat),
            tripletex_client=tripletex_client,
        )

        credit_note = tripletex_client.put(
            f"/invoice/{invoice['id']}/:createCreditNote",
            params={
                "date": invoice["invoiceDate"]
            },  # TODO: i might want to not send to customer
        ).json()
        credit_note


if __name__ == "__main__":
    pass
    # for prompt in prompts:
    #     res = CreditNote.parse(prompt)
    #     print(f"{prompt=}")
    #     print(f"{res=}")
    #     print()
    #     break
