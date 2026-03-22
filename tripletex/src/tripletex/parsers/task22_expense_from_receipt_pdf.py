from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import get_department_by_name
from tripletex.my_log import configure_logging

prompts = [
    "Vi trenger USB-hub fra denne kvitteringen bokfort pa avdeling HR. Bruk riktig utgiftskonto basert pa kjopet, og sorg for korrekt MVA-behandling.",
    "We need the Togbillett expense from this receipt posted to department Produksjon. Use the correct expense account and ensure proper VAT treatment.",
    "Vi treng Oppbevaringsboks fra denne kvitteringa bokfort pa avdeling Regnskap. Bruk rett utgiftskonto basert pa kjopet, og sorg for korrekt MVA-behandling.",
    "We need the Kontorstoler expense from this receipt posted to department Økonomi. Use the correct expense account and ensure proper VAT treatment.",
    "Precisamos da despesa de Kaffemøte deste recibo registada no departamento Salg. Use a conta de despesas correta e garanta o tratamento correto do IVA.",
    "Vi treng USB-hub fra denne kvitteringa bokfort pa avdeling Kvalitetskontroll. Bruk rett utgiftskonto basert pa kjopet, og sorg for korrekt MVA-behandling.",
    "Nous avons besoin de la depense Oppbevaringsboks de ce recu enregistree au departement Produksjon. Utilisez le bon compte de charges et assurez le traitement correct de la TVA.",
    "Wir benotigen die Headset-Ausgabe aus dieser Quittung in der Abteilung Salg. Verwenden Sie das richtige Aufwandskonto und stellen Sie die korrekte MwSt.-Behandlung sicher.",
    "Necesitamos el gasto de USB-hub de este recibo registrado en el departamento Utvikling. Usa la cuenta de gastos correcta y asegura el tratamiento correcto del IVA.",
    "Precisamos da despesa de Overnatting deste recibo registada no departamento Utvikling. Use a conta de despesas correta e garanta o tratamento correto do IVA.",
    'Wir benotigen die Togbillett-Ausgabe aus dieser Quittung in der Abteilung Logistikk. Verwenden Sie das richtige Aufwandskonto und stellen Sie die korrekte MwSt.-Behandlung sicher.',
    'Wir benotigen die Skrivebordlampe-Ausgabe aus dieser Quittung in der Abteilung Produksjon. Verwenden Sie das richtige Aufwandskonto und stellen Sie die korrekte MwSt.-Behandlung sicher.',
    'Vi trenger Togbillett fra denne kvitteringen bokfort pa avdeling Administrasjon. Bruk riktig utgiftskonto basert pa kjopet, og sorg for korrekt MVA-behandling.',
    'Nous avons besoin de la depense Skrivebordlampe de ce recu enregistree au departement Kvalitetskontroll. Utilisez le bon compte de charges et assurez le traitement correct de la TVA.',
]

attachments = [
    "data/files/parsed/kvittering_en_03.txt",
    "data/files/parsed/kvittering_de_06.txt",
    "data/files/parsed/kvittering_nn_03.txt",
    "data/files/parsed/kvittering_nn_06.txt",
    "data/files/parsed/kvittering_pt_05.txt",
    "data/files/parsed/kvittering_es_03.txt",
    "data/files/parsed/kvittering_fr_08.txt",
    "data/files/parsed/kvittering_pt_08.txt",
]

logger = configure_logging()

EXPENSE_ACCOUNT_SYSTEM_PROMPT = """You are parsing a Norwegian receipt and expense posting request.
Extract the fields requested. For expense_account_number, use the Norwegian standard chart of accounts:
- USB-hub, Headset, Tastatur, Mus, Datautstyr → 6551 (Datautstyr hardware)
- Kontorstoler, Inventar → 6540 (Inventar)
- Oppbevaringsboks, Kontorrekvisita, Rekvisita → 6800 (Kontorrekvisita)
- Kaffemøte, Forretningslunsj, Middag representasjon → 7350 (Representasjon, fradragsberettiget)
- Togbillett, Flybillett → 7140 (Reisekostnad, ikke oppgavepliktig)
- Overnatting → 7130 (Reisekostnad, oppgavepliktig)
Pick the most fitting account number from the list above."""


class ExpenseFromReceiptPDF(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expense_item_name: str
    department_name: str
    expense_account_number: Annotated[int, Field(description="Norwegian chart of accounts number (e.g. 6551, 6540, 6800, 7350, 7140, 7130)")]
    item_price_incl_vat: Annotated[float, Field(description="Price of the specific item including VAT (item_price * 1.25)")]
    item_price_excl_vat: Annotated[float, Field(description="Price of the specific item excluding VAT (as listed on receipt)")]
    receipt_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Receipt date in YYYY-MM-DD format")]

    @classmethod
    def parse(cls, prompt: str, attachment: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=EXPENSE_ACCOUNT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n--- RECEIPT ---\n{attachment}",
                }
            ],
            output_format=ExpenseFromReceiptPDF,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - The department named in the prompt already exists.
        #   - Standard chart of accounts is present (expense accounts 6xxx/7xxx,
        #     bank account 1920).
        #   - VAT is always 25% (vatType id=1, "Fradrag inngående avgift, høy sats").
        #   - Receipt prices listed are excl. VAT; total on receipt is incl. VAT.
        #
        # Assumptions about what scoring checks:
        #   - A voucher exists with the correct date (receipt date).
        #   - Expense posting: correct account, correct net amount, correct department.
        #   - VAT posting: auto-generated by Tripletex from vatType=1.
        #   - Bank posting: correct gross amount on account 1920.
        #   - Voucher description may or may not be checked (we use the item name).
        # ── GET department (always pre-exists) ───────────────────────────
        dept = get_department_by_name(self.department_name, tripletex_client)
        department_id = dept["id"]
        logger.info(
            "task22.department.found",
            extra={"department_id": department_id, "dept_name": dept["name"]},
        )

        # ── GET ledger accounts ──────────────────────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}
        logger.info(
            "task22.accounts.fetched", extra={"count": len(accts)}
        )

        expense_account = accts[self.expense_account_number]
        bank_account = accts[1920]  # Bankinnskudd
        logger.info(
            "task22.accounts.resolved",
            extra={
                "expense_account": expense_account,
                "bank_account": bank_account,
            },
        )

        # ── POST voucher (1 write call) ──────────────────────────────────
        # Row 1: debit expense account (net amount + VAT type 1 for 25% input VAT)
        # Row 2: credit bank account (gross amount)
        # Row 0: system auto-generates the VAT posting
        voucher_payload = {
            "date": self.receipt_date,
            "description": self.expense_item_name,
            "postings": [
                {
                    "account": {"id": expense_account["id"]},
                    "amountCurrency": self.item_price_excl_vat,
                    "amountGross": self.item_price_incl_vat,
                    "amountGrossCurrency": self.item_price_incl_vat,
                    "vatType": {"id": 1},  # 25% fradrag inngående avgift
                    "department": {"id": department_id},
                    "row": 1,
                },
                {
                    "account": {"id": bank_account["id"]},
                    "amountCurrency": -self.item_price_incl_vat,
                    "amountGross": -self.item_price_incl_vat,
                    "amountGrossCurrency": -self.item_price_incl_vat,
                    "row": 2,
                },
            ],
        }
        logger.info("task22.voucher.creating", extra={"payload": voucher_payload})
        voucher_r = tripletex_client.post("/ledger/voucher", json=voucher_payload)
        logger.info(
            "task22.voucher.response",
            extra={"status": voucher_r.status_code, "body": voucher_r.json()},
        )
        voucher_r.raise_for_status()
        voucher_data = voucher_r.json()
        logger.info("task22.voucher.created", extra={"response": voucher_data})

        logger.info(
            "task22.completed",
            extra={
                "voucher_id": voucher_data["value"]["id"],
                "department_id": department_id,
                "expense_account": self.expense_account_number,
                "amount_excl_vat": self.item_price_excl_vat,
                "amount_incl_vat": self.item_price_incl_vat,
            },
        )


if __name__ == "__main__":
    from pathlib import Path

    from tripletex.herman_tasks.utils import TripletexCredentials

    tripletex_client = TripletexCredentials.placeholder_TODO().to_client()

    prompt = prompts[0]
    attachment = Path(attachments[0]).read_text()

    parsed = ExpenseFromReceiptPDF.parse(prompt, attachment)
    logger.info("task22.parsed", extra={"parsed": parsed.model_dump()})

    parsed.solve(tripletex_client=tripletex_client)
