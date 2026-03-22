from typing import Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, EmailStr

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_current_year_month_day_utc,
    get_employee,
)

prompts = [
    'Registrer ei reiserekning for Marit Vik (marit.vik@example.org) for "Konferanse Kristiansand". Reisa varte 4 dagar med diett (dagssats 800 kr). Utlegg: flybillett 7600 kr og taxi 600 kr.',
    'Registrer ei reiserekning for Svein Berge (svein.berge@example.org) for "Kundebesøk Trondheim". Reisa varte 5 dagar med diett (dagssats 800 kr). Utlegg: flybillett 2850 kr og taxi 200 kr.',
    'Registrer ei reiserekning for Svein Vik (svein.vik@example.org) for "Kundebesøk Bergen". Reisa varte 5 dagar med diett (dagssats 800 kr). Utlegg: flybillett 7900 kr og taxi 250 kr.',
    'Registrer en reiseregning for Ingrid Larsen (ingrid.larsen@example.org) for "Kundebesøk Trondheim". Reisen varte 2 dager med diett (dagsats 800 kr). Utlegg: flybillett 2500 kr og taxi 600 kr.',
    'Registrer en reiseregning for Magnus Haugen (magnus.haugen@example.org) for "Kundebesøk Bergen". Reisen varte 4 dager med diett (dagsats 800 kr). Utlegg: flybillett 5050 kr og taxi 750 kr.',
    'Register a travel expense for Charlotte Williams (charlotte.williams@example.org) for "Client visit Bodø". The trip lasted 3 days with per diem (daily rate 800 NOK). Expenses: flight ticket 6200 NOK and taxi 400 NOK.',
    'Register a travel expense for William Wilson (william.wilson@example.org) for "Client visit Trondheim". The trip lasted 2 days with per diem (daily rate 800 NOK). Expenses: flight ticket 7600 NOK and taxi 700 NOK.',
    'Erfassen Sie eine Reisekostenabrechnung für Johanna Hoffmann (johanna.hoffmann@example.org) für "Kundenbesuch Tromsø". Die Reise dauerte 3 Tage mit Tagegeld (Tagessatz 800 NOK). Auslagen: Flugticket 5250 NOK und Taxi 250 NOK.',
]


class TravelExpense(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str
    last_name: str
    email: EmailStr
    trip_description: str
    trip_destination: str
    days: int
    daily_rate: float
    flight_cost: float
    taxi_cost: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=TravelExpense,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        employee = get_employee(employee_email=self.email, tripletex_client=tripletex_client)

        (payment_type,) = tripletex_client.get("/travelExpense/paymentType").json()["values"]
        payment_type

        travel_expense_categories = tripletex_client.get(
            "/travelExpense/costCategory"
        ).json()["values"]

        (flight_travel_expense_category,) = [
            x for x in travel_expense_categories if x["displayName"] == "Fly"
        ]
        (taxi_expense_category,) = [
            x for x in travel_expense_categories if x["displayName"] == "Taxi"
        ]

        travel_expense_rates = tripletex_client.get(
            "/travelExpense/rate",
            params={"type": "PER_DIEM", "isValidDomestic": True},
        ).json()["values"]

        (travel_expense_rates_filtered_by_rate,) = [
            x for x in travel_expense_rates if x["rate"] == self.daily_rate
        ]

        json_data = {
            "employee": {"id": employee["id"]},
            "title": self.trip_description,
            "travelDetails": {
                "isForeignTravel": False,
                "isDayTrip": False,
                "departureDate": get_current_year_month_day_utc(
                    days_offset_forward=-self.days
                ),
                "returnDate": get_current_year_month_day_utc(),
                "purpose": self.trip_description,
            },
            "isChargeable": False,
            "isFixedInvoicedAmount": False,
            "isIncludeAttachedReceiptsWhenReinvoicing": False,
            "perDiemCompensations": [
                {
                    "rateType": {"id": travel_expense_rates_filtered_by_rate["id"]},
                    "count": self.days,
                    "location": self.trip_description,
                }
            ],
            "costs": [
                {
                    "paymentType": {"id": payment_type["id"]},
                    "date": get_current_year_month_day_utc(
                        days_offset_forward=-self.days
                    ),
                    "costCategory": {"id": flight_travel_expense_category["id"]},
                    "amountCurrencyIncVat": self.flight_cost,
                    "comments": "flight ticket",
                },
                {
                    "paymentType": {"id": payment_type["id"]},
                    "date": get_current_year_month_day_utc(
                        days_offset_forward=-self.days
                    ),
                    "costCategory": {"id": taxi_expense_category["id"]},
                    "amountCurrencyIncVat": self.taxi_cost,
                    "comments": "taxi",
                },
            ],
        }
        json_data

        travel_expense = tripletex_client.post("/travelExpense", json=json_data).json()
        travel_expense


if __name__ == "__main__":
    pass
    # for prompt in prompts:
    #     res = TravelExpense.parse(prompt)
    #     print(f"{prompt=}")
    #     print(f"{res=}")
    #     print()
    #     break
