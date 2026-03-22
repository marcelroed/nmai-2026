from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "Crea y envía una factura al cliente Luna SL (org. nº 844920520) por 20200 NOK sin IVA. La factura es por Servicio de red.",
    "Opprett og send en faktura til kunden Polaris AS (org.nr 963373937) på 32600 kr eksklusiv MVA. Fakturaen gjelder Webdesign.",
    "Create and send an invoice to the customer Clearwater Ltd (org no. 935400759) for 3100 NOK excluding VAT. The invoice is for Data Advisory.",
    "Crie e envie uma fatura ao cliente Porto Alegre Lda (org. nº 826870192) por 22700 NOK sem IVA. A fatura refere-se a Design web.",
    "Crea y envía una factura al cliente Montaña SL (org. nº 831306742) por 48600 NOK sin IVA. La factura es por Licencia de software.",
    'Erstellen und senden Sie eine Rechnung an den Kunden Bergwerk GmbH (Org.-Nr. 868341580) über 18200 NOK ohne MwSt. Die Rechnung betrifft Analysebericht.',
]


class CreateAndSendInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    amount_excl_vat: float
    product_description: str

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreateAndSendInvoice,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = CreateAndSendInvoice.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
