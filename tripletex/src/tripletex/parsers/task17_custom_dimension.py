from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_current_year_month_day_utc,
)
from tripletex.my_log import configure_logging

prompts = [
    'Cree una dimensión contable personalizada "Produktlinje" con los valores "Basis" y "Premium". Luego registre un asiento en la cuenta 7140 por 39600 NOK, vinculado al valor de dimensión "Premium".',
    'Opprett ein fri rekneskapsdimensjon "Marked" med verdiane "Offentlig" og "Privat". Bokfør deretter eit bilag på konto 7300 for 49300 kr, knytt til dimensjonsverdien "Offentlig".',
    'Opprett ein fri rekneskapsdimensjon "Prosjekttype" med verdiane "Utvikling" og "Internt". Bokfør deretter eit bilag på konto 6860 for 40600 kr, knytt til dimensjonsverdien "Utvikling".',
    'Erstellen Sie eine benutzerdefinierte Buchhaltungsdimension "Kostsenter" mit den Werten "Kundeservice" und "Markedsføring". Buchen Sie dann einen Beleg auf Konto 7100 über 42400 NOK, verknüpft mit dem Dimensionswert "Markedsføring".',
    'Erstellen Sie eine benutzerdefinierte Buchhaltungsdimension "Prosjekttype" mit den Werten "Internt" und "Utvikling". Buchen Sie dann einen Beleg auf Konto 6340 über 44500 NOK, verknüpft mit dem Dimensionswert "Internt".',
    'Opprett ein fri rekneskapsdimensjon "Produktlinje" med verdiane "Basis" og "Avansert". Bokfør deretter eit bilag på konto 6340 for 15000 kr, knytt til dimensjonsverdien "Avansert".',
    'Opprett en fri regnskapsdimensjon "Prosjekttype" med verdiene "Forskning" og "Utvikling". Bokfør deretter et bilag på konto 6590 for 10800 kr, knyttet til dimensjonsverdien "Forskning".',
]

logger = configure_logging()


class CustomDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_name: str
    dimension_values: list[str]
    account_number: Annotated[
        int, Field(description="Account number for the voucher posting")
    ]
    amount: float
    linked_dimension_value: str

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CustomDimension,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - No custom dimensions exist yet (fresh sandbox, max 3 allowed).
        #   - Always exactly 2 dimension values to create.
        #   - The voucher links to one of the two values via freeAccountingDimension{N}
        #     where N = the dimensionIndex returned when creating the dimension.
        #
        # Assumptions about what scoring checks:
        #   - Dimension exists with the correct name.
        #   - Both dimension values exist.
        #   - A voucher exists on the correct account for the correct amount,
        #     linked to the specified dimension value.
        today = get_current_year_month_day_utc()

        # ── GET ledger accounts ──────────────────────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}

        # ── WRITE 1: create accounting dimension ─────────────────────────
        dim_payload = {"dimensionName": self.dimension_name}
        logger.info("task17.dimension.creating", extra={"payload": dim_payload})
        dim_r = tripletex_client.post(
            "/ledger/accountingDimensionName", json=dim_payload
        )
        logger.info(
            "task17.dimension.response",
            extra={"status": dim_r.status_code, "body": dim_r.json()},
        )
        dim_r.raise_for_status()
        dim_data = dim_r.json()["value"]
        dim_id = dim_data["id"]
        dim_index = dim_data["dimensionIndex"]
        logger.info(
            "task17.dimension.created",
            extra={"dim_id": dim_id, "dim_index": dim_index},
        )

        # ── WRITE 2-3: create dimension values ──────────────────────────
        linked_value_id = None
        for val_name in self.dimension_values:
            val_payload = {
                "dimensionName": {"id": dim_id},
                "name": val_name,
            }
            logger.info("task17.value.creating", extra={"payload": val_payload})
            val_r = tripletex_client.post(
                "/ledger/accountingDimensionValue", json=val_payload
            )
            logger.info(
                "task17.value.response",
                extra={"status": val_r.status_code, "body": val_r.json()},
            )
            val_r.raise_for_status()
            val_data = val_r.json()["value"]
            if val_name == self.linked_dimension_value:
                linked_value_id = val_data["id"]

        assert linked_value_id is not None, (
            f"Linked dimension value '{self.linked_dimension_value}' not found in {self.dimension_values}"
        )

        # ── WRITE 4: post voucher linked to dimension value ──────────────
        dim_field = f"freeAccountingDimension{dim_index}"
        voucher_payload = {
            "date": today,
            "description": f"{self.dimension_name}: {self.linked_dimension_value}",
            "postings": [
                {
                    "account": {"id": accts[self.account_number]["id"]},
                    "amountCurrency": self.amount,
                    "amountGross": self.amount,
                    "amountGrossCurrency": self.amount,
                    dim_field: {"id": linked_value_id},
                    "row": 1,
                },
                {
                    "account": {"id": accts[1920]["id"]},
                    "amountCurrency": -self.amount,
                    "amountGross": -self.amount,
                    "amountGrossCurrency": -self.amount,
                    "row": 2,
                },
            ],
        }
        logger.info("task17.voucher.creating", extra={"payload": voucher_payload})
        v_r = tripletex_client.post("/ledger/voucher", json=voucher_payload)
        logger.info(
            "task17.voucher.response",
            extra={"status": v_r.status_code, "body": v_r.json()},
        )
        v_r.raise_for_status()

        logger.info("task17.completed")


if __name__ == "__main__":
    for prompt in prompts:
        res = CustomDimension.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
