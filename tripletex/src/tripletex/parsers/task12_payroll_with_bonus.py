from typing import Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_current_year_month_day_utc,
    get_employee,
)
from tripletex.my_log import configure_logging

logger = configure_logging()

prompts = [
    "Run payroll for William Taylor (william.taylor@example.org) for this month. The base salary is 39400 NOK. Add a one-time bonus of 11800 NOK on top of the base salary. If the salary API is unavailable, you can use manual vouchers on salary accounts (5000-series) to record the payroll expense.",
    "Køyr løn for Brita Stølsvik (brita.stlsvik@example.org) for denne månaden. Grunnløn er 36000 kr. Legg til ein eingongsbonus på 15400 kr i tillegg til grunnløna.",
    "Køyr løn for Eirik Lunde (eirik.lunde@example.org) for denne månaden. Grunnløn er 39100 kr. Legg til ein eingongsbonus på 8200 kr i tillegg til grunnløna.",
    "Køyr løn for Eirik Brekke (eirik.brekke@example.org) for denne månaden. Grunnløn er 41050 kr. Legg til ein eingongsbonus på 9800 kr i tillegg til grunnløna.",
    "Processe o salário de Rafael Silva (rafael.silva@example.org) para este mês. O salário base é de 41150 NOK. Adicione um bónus único de 8500 NOK além do salário base.",
    "Kjør lønn for Tor Bakken (tor.bakken@example.org) for denne måneden. Grunnlønn er 34100 kr. Legg til en engangsbonus på 6550 kr i tillegg til grunnlønnen. Dersom lønns-API-et ikke fungerer, kan du bruke manuelle bilag på lønnskontoer (5000-serien) for å registrere lønnskostnaden.",
    "Führen Sie die Gehaltsabrechnung für Sophia Müller (sophia.muller@example.org) für diesen Monat durch. Das Grundgehalt beträgt 48350 NOK. Fügen Sie einen einmaligen Bonus von 15450 NOK zum Grundgehalt hinzu.",
    "Exécutez la paie de Sarah Moreau (sarah.moreau@example.org) pour ce mois. Le salaire de base est de 56900 NOK. Ajoutez une prime unique de 15800 NOK en plus du salaire de base.",
    "Ejecute la nómina de Fernando Sánchez (fernando.sanchez@example.org) para este mes. El salario base es de 42350 NOK. Añada una bonificación única de 12850 NOK además del salario base.",
    'Processe o salário de Lucas Martins (lucas.martins@example.org) para este mês. O salário base é de 39000 NOK. Adicione um bónus único de 10500 NOK além do salário base.',
    'Run payroll for Daniel Smith (daniel.smith@example.org) for this month. The base salary is 54850 NOK. Add a one-time bonus of 6800 NOK on top of the base salary. If the salary API is unavailable, you can use manual vouchers on salary accounts (5000-series) to record the payroll expense.',
    'Exécutez la paie de Louis Richard (louis.richard@example.org) pour ce mois. Le salaire de base est de 36600 NOK. Ajoutez une prime unique de 19050 NOK en plus du salaire de base.',
]


class PayrollWithBonus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str
    email: EmailStr
    base_salary: float
    bonus_amount: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=PayrollWithBonus,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        # TODO: add logging
        employee = get_employee(employee_email=self.email, tripletex_client=tripletex_client)

        salary_type = tripletex_client.get("/salary/type").json()["values"]
        logger.info("got the salary types", extra={"salary_type": salary_type})

        (fixed_salary_type,) = [x for x in salary_type if x["name"] == "Fastlønn"]
        (bonus_salary_type,) = [x for x in salary_type if x["name"] == "Bonus"]

        date = get_current_year_month_day_utc()
        year = int(get_current_year_month_day_utc()[:4])
        month = int(get_current_year_month_day_utc()[5:7])
        json_data = {
            "date": date,
            "year": year,
            "month": month,
            "payslips": [
                {
                    "employee": {"id": employee["id"]},
                    "date": get_current_year_month_day_utc(),
                    "year": year,
                    "month": month,
                    "specifications": [
                        {
                            "salaryType": {"id": fixed_salary_type["id"]},
                            "year": year,
                            "month": month,
                            "count": 1,
                            "rate": self.base_salary,
                        },
                        {
                            "salaryType": {"id": bonus_salary_type["id"]},
                            "year": year,
                            "month": month,
                            "count": 1,
                            "rate": self.bonus_amount,
                        },
                    ],
                }
            ],
        }
        json_data
        salary_post_r = tripletex_client.post("/salary/transaction", json=json_data).json()
        salary_post_r


if __name__ == "__main__":
    # for prompt in prompts:
    #     res = PayrollWithBonus.parse(prompt)
    #     print(f"{prompt=}")
    #     print(f"{res=}")
    #     print()
    pass

    #
    # tripletex_creds = TripletexCredentials.placeholder_TODO()
    # self = PayrollWithBonus(
    #     first_name="William",
    #     last_name="Taylor",
    #     email="fernando.garcia@example.org",
    #     base_salary=39400.0,
    #     bonus_amount=11800.0,
    # )
    # self.solve(TripletexCredentials.placeholder_TODO())
