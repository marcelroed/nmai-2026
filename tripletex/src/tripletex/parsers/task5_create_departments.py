from typing import Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict

from tripletex.client import LoggedHTTPClient

prompts = [
    'Crie três departamentos no Tripletex: "Drift", "Kundeservice" e "HR".',
    'Opprett tre avdelinger i Tripletex: "Utvikling", "Drift" og "HR".',
    'Créez trois départements dans Tripletex : "Produksjon", "Salg" et "Logistikk".',
    'Créez trois départements dans Tripletex : "Administrasjon", "IT" et "Utvikling".',
    'Créez trois départements dans Tripletex : "Produksjon", "Salg" et "Innkjøp".',
    'Crea tres departamentos en Tripletex: "Logistikk", "Produksjon" y "Drift".',
    'Créez trois départements dans Tripletex : "Økonomi", "Lager" et "IT".',
    'Crea tres departamentos en Tripletex: "IT", "Drift" y "Kundeservice".',
    'Crie três departamentos no Tripletex: "Kundeservice", "Innkjøp" e "Regnskap".',
    'Create three departments in Tripletex: "Produksjon", "Kundeservice", and "Økonomi".',
]


class CreateDepartments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    department_names: list[str]

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreateDepartments,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = CreateDepartments.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
