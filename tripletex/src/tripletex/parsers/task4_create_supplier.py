from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "Registrieren Sie den Lieferanten Brückentor GmbH mit der Organisationsnummer 867835989. E-Mail: faktura@brckentorgmbh.no.",
    "Registrieren Sie den Lieferanten Waldstein GmbH mit der Organisationsnummer 891505019. E-Mail: faktura@waldsteingmbh.no.",
    "Registe o fornecedor Floresta Lda com número de organização 883568885. E-mail: faktura@florestalda.no.",
    "Registrer leverandøren Fossekraft AS med organisasjonsnummer 977371635. E-post: faktura@fossekraft.no.",
    "Registe o fornecedor Luz do Sol Lda com número de organização 894596554. E-mail: faktura@luzdosollda.no.",
    "Registrer leverandøren Sjøbris AS med organisasjonsnummer 811212717. E-post: faktura@sjbris.no.",
    "Registe o fornecedor Oceano Lda com número de organização 841149394. E-mail: faktura@oceanolda.no.",
    "Register the supplier Silveroak Ltd with organization number 889586605. Email: faktura@silveroakltd.no.",
    'Registrer leverandøren Bergvik AS med organisasjonsnummer 910473166. E-post: faktura@bergvik.no.',
    'Registrer leverandøren Vestfjord AS med organisasjonsnummer 914908787. E-post: faktura@vestfjord.no.',
]


class CreateSupplier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    email: EmailStr

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreateSupplier,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = CreateSupplier.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
