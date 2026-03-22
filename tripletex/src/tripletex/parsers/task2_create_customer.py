from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "Opprett kunden Snøhetta AS med organisasjonsnummer 969719878. Adressen er Industriveien 148, 2317 Hamar. E-post: post@snhetta.no.",
    "Crea el cliente Río Verde SL con número de organización 919234830. La dirección es Solveien 5, 4006 Stavanger. Correo: post@rio.no.",
    "Opprett kunden Nordhav AS med organisasjonsnummer 980461912. Adressen er Solveien 7, 4006 Stavanger. E-post: post@nordhav.no.",
    "Créez le client Colline SARL avec le numéro d'organisation 939137599. L'adresse est Kirkegata 77, 4611 Kristiansand. E-mail : post@colline.no.",
    "Crie o cliente Porto Alegre Lda com número de organização 834147254. O endereço é Storgata 65, 4611 Kristiansand. E-mail: post@porto.no.",
    'Opprett kunden Bølgekraft AS med organisasjonsnummer 812297848. Adressa er Havnegata 113, 7010 Trondheim. E-post: post@blgekraft.no.',
    'Create the customer Northwave Ltd with organization number 964179239. The address is Nygata 39, 2317 Hamar. Email: post@northwave.no.',
]


class CreateCustomer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    street_address: str
    postal_code: Annotated[str, Field(pattern=r"^\d{4}$")]
    city: str
    email: EmailStr

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreateCustomer,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = CreateCustomer.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
