from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "Crie um pedido para o cliente Estrela Lda (org. nº 842487803) com os produtos Design web (1851) a 24050 NOK e Consultoria de dados (5065) a 13450 NOK. Converta o pedido em fatura e registe o pagamento total.",
    "Opprett en ordre for kunden Fjordkraft AS (org.nr 911511053) med produktene Opplæring (7579) til 14650 kr og Webdesign (2292) til 11800 kr. Konverter ordren til faktura og registrer full betaling.",
    "Crie um pedido para o cliente Horizonte Lda (org. nº 904130338) com os produtos Serviço de rede (6247) a 15250 NOK e Desenvolvimento de sistemas (5919) a 13250 NOK. Converta o pedido em fatura e registe o pagamento total.",
    "Crea un pedido para el cliente Río Verde SL (org. nº 937237243) con los productos Informe de análisis (5700) a 33200 NOK y Diseño web (2680) a 17200 NOK. Convierte el pedido en factura y registra el pago completo.",
    "Crea un pedido para el cliente Dorada SL (org. nº 984411359) con los productos Desarrollo de sistemas (5240) a 21950 NOK y Sesión de formación (5871) a 6350 NOK. Convierte el pedido en factura y registra el pago completo.",
    "Opprett ein ordre for kunden Bølgekraft AS (org.nr 908252764) med produkta Nettverksteneste (6065) til 25500 kr og Systemutvikling (2511) til 23550 kr. Konverter ordren til faktura og registrer full betaling.",
    "Erstellen Sie einen Auftrag für den Kunden Waldstein GmbH (Org.-Nr. 899060113) mit den Produkten Netzwerkdienst (5411) zu 29200 NOK und Schulung (7883) zu 10350 NOK. Wandeln Sie den Auftrag in eine Rechnung um und registrieren Sie die vollständige Zahlung.",
    "Crea un pedido para el cliente Río Verde SL (org. nº 951612936) con los productos Informe de análisis (4430) a 20900 NOK y Sesión de formación (7773) a 18350 NOK. Convierte el pedido en factura y registra el pago completo.",
    "Opprett en ordre for kunden Stormberg AS (org.nr 870531559) med produktene Vedlikehold (4665) til 35200 kr og Systemutvikling (7431) til 4400 kr. Konverter ordren til faktura og registrer full betaling.",
]


class OrderProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    product_number: Annotated[str, Field(pattern=r"^\d+$")]
    price: float


class OrderToInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    products: list[OrderProduct]

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=OrderToInvoice,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = OrderToInvoice.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
