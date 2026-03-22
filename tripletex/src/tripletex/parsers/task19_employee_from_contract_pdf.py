from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "Vous avez recu un contrat de travail (voir PDF ci-joint). Creez l'employe dans Tripletex avec tous les details du contrat : numero d'identite nationale, date de naissance, departement, code de profession, salaire, pourcentage d'emploi et date de debut.",
    "Du har motteke ein arbeidskontrakt (sjaa vedlagt PDF). Opprett den tilsette i Tripletex med alle detaljar fraa kontrakten: personnummer, fodselsdato, avdeling, stillingskode, lonn, stillingsprosent og startdato.",
    "Voce recebeu um contrato de trabalho (ver PDF anexo). Crie o funcionario no Tripletex com todos os detalhes do contrato: numero de identidade nacional, data de nascimento, departamento, codigo de ocupacao, salario, percentagem de emprego e data de inicio.",
    "You received an employment contract (see attached PDF). Create the employee in Tripletex with all details from the contract: national identity number, date of birth, department, occupation code, salary, employment percentage, and start date.",
    "Sie haben einen Arbeitsvertrag erhalten (siehe beigefugte PDF). Erstellen Sie den Mitarbeiter in Tripletex mit allen Details aus dem Vertrag: Personalnummer, Geburtsdatum, Abteilung, Berufsschluessel, Gehalt, Beschaeftigungsprozentsatz und Startdatum.",
]

attachments = [
    "data/files/parsed/arbeidskontrakt_en_01.txt",
    "data/files/parsed/arbeidskontrakt_nn_05.txt",
    "data/files/parsed/arbeidskontrakt_fr_03.txt",
    "data/files/parsed/arbeidskontrakt_pt_02.txt",
    "data/files/parsed/arbeidskontrakt_en_08.txt",
    "data/files/parsed/arbeidskontrakt_fr_06.txt",
]


class EmployeeFromContractPDF(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str
    birth_year: int
    birth_month: int
    birth_day: int
    national_identity_number: Annotated[str, Field(pattern=r"^\d{11}$")]
    email: EmailStr
    bank_account_number: str
    department: str
    occupation_code: Annotated[str, Field(pattern=r"^\d{4}$")]
    employment_percentage: float
    annual_salary: float
    start_year: int
    start_month: int
    start_day: int

    @classmethod
    def parse(cls, prompt: str, attachment: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n--- ATTACHMENT ---\n{attachment}",
                }
            ],
            output_format=EmployeeFromContractPDF,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        department_create_r = tripletex_client.post(
            "/department", json={"name": self.department}
        ).json()
        department_create_r

        employee_payload = {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "dateOfBirth": f"{self.birth_year}-{self.birth_month:02}-{self.birth_day:02}",
            "email": self.email,
            "nationalIdentityNumber": self.national_identity_number,
            "bankAccountNumber": self.bank_account_number,
            "department": {"id": department_create_r["value"]["id"]},
            "userType": "STANDARD",
        }
        employee_payload

        employee_post_r = tripletex_client.post("/employee", json=employee_payload).json()
        employee_post_r

        employee_employment_payload = {
            "employee": {"id": employee_post_r["value"]["id"]},
            "division": {"id": department_create_r["value"]["id"]},
            "startDate": f"{self.start_year}-{self.start_month:02}-{self.start_day:02}",
            "isMainEmployer": True,
            "employmentDetails": [
                {
                    "date": f"{self.start_year}-{self.start_month:02}-{self.start_day:02}",
                    "employmentType": "ORDINARY",
                    "employmentForm": "PERMANENT",
                    "remunerationType": "MONTHLY_WAGE",
                    "occupationCode": {"id": self.occupation_code},
                    "percentageOfFullTimeEquivalent": self.employment_percentage,
                    "annualSalary": self.annual_salary,
                }
            ],
        }
        employee_employment_payload

        employee_employment_r = tripletex_client.post(
            "/employee/employment", json=employee_employment_payload
        )
        employee_employment_r.json()


if __name__ == "__main__":
    pass
    # from pathlib import Path
    #
    # for attachment_path in attachments:
    #     attachment = Path(attachment_path).read_text()
    #     res = EmployeeFromContractPDF.parse(prompts[0], attachment)
    #     print(f"file={attachment_path}")
    #     print(f"{res=}")
    #     print()
    #     break
