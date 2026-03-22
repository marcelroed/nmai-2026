from typing import Annotated, Literal, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "Me har ein ny tilsett som heiter Torbjørn Neset, fødd 14. November 1991. Opprett vedkomande som tilsett med e-post torbjrn.neset@example.org og startdato 11. February 2026.",
    "Vi har en ny ansatt som heter Karin Berg, født 11. March 1992. Opprett vedkommende som ansatt med e-post karin.berg@example.org og startdato 29. March 2026.",
    "Me har ein ny tilsett som heiter Geir Stølsvik, fødd 6. March 1990. Opprett vedkomande som tilsett med e-post geir.stlsvik@example.org og startdato 14. November 2026.",
    "Nous avons un nouvel employé nommé Manon Thomas, né le 14. December 1981. Veuillez le créer en tant qu'employé avec l'e-mail manon.thomas@example.org et la date de début 22. January 2026.",
    "Tenemos un nuevo empleado llamado Lucía Rodríguez, nacido el 30. April 1996. Créelo como empleado con el correo lucia.rodriguez@example.org y fecha de inicio 22. April 2026.",
    "Vi har en ny ansatt som heter Lars Berg, født 10. April 1995. Opprett vedkommende som ansatt med e-post lars.berg@example.org og startdato 16. November 2026.",
    "Me har ein ny tilsett som heiter Gunnhild Eide, fødd 21. June 1997. Opprett vedkomande som tilsett med e-post gunnhild.eide@example.org og startdato 28. June 2026.",
]


class CreateEmployee(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str
    birth_year: int
    birth_month: int
    birth_day: int
    email: EmailStr
    start_year: int
    start_month: int
    start_day: int

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CreateEmployee,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = CreateEmployee.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
