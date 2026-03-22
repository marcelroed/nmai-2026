from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    'El cliente Viento SL (org. nº 908616537) tiene una factura pendiente de 37850 NOK sin IVA por "Mantenimiento". Registre el pago completo de esta factura.',
    'The customer Ironbridge Ltd (org no. 985423849) has an outstanding invoice for 32900 NOK excluding VAT for "Web Design". Register full payment on this invoice.',
    'Der Kunde Sonnental GmbH (Org.-Nr. 855482207) hat eine offene Rechnung über 33500 NOK ohne MwSt. für "Wartung". Registrieren Sie die vollständige Zahlung dieser Rechnung.',
    'Kunden Strandvik AS (org.nr 840390055) har ein uteståande faktura på 27050 kr eksklusiv MVA for "Datarådgjeving". Registrer full betaling på denne fakturaen.',
    'El cliente Costa Brava SL (org. nº 833355937) tiene una factura pendiente de 45700 NOK sin IVA por "Asesoría de datos". Registre el pago completo de esta factura.',
    'Le client Forêt SARL (nº org. 925519685) a une facture impayée de 40750 NOK hors TVA pour "Design web". Enregistrez le paiement intégral de cette facture.',
    'El cliente Solmar SL (org. nº 939332235) tiene una factura pendiente de 46700 NOK sin IVA por "Diseño web". Registre el pago completo de esta factura.',
    'El cliente Solmar SL (org. nº 866440034) tiene una factura pendiente de 30000 NOK sin IVA por "Almacenamiento en la nube". Registre el pago completo de esta factura.',
]


class RegisterPayment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    amount_excl_vat: float
    invoice_description: str

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=RegisterPayment,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = RegisterPayment.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
