from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    'Opprett produktet "Frokostblanding" med produktnummer 1391. Prisen er 37450 kr eksklusiv MVA, og MVA-sats for næringsmidler på 15 % skal brukes.',
    'Crie o produto "Serviço de rede" com número de produto 4252. O preço é 8200 NOK sem IVA, utilizando a taxa padrão de 25 %.',
    'Opprett produktet "Analyserapport" med produktnummer 3637. Prisen er 31900 kr eksklusiv MVA, og standard MVA-sats på 25 % skal nyttast.',
    'Opprett produktet "Datarådgjeving" med produktnummer 4993. Prisen er 16250 kr eksklusiv MVA, og standard MVA-sats på 25 % skal nyttast.',
    'Create the product "Training Session" with product number 7908. The price is 26250 NOK excluding VAT, using the standard 25% VAT rate.',
]


class CreateProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    product_number: Annotated[str, Field(pattern=r"^\d+$")]
    price_excl_vat: float
    vat_rate: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreateProduct,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = CreateProduct.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
