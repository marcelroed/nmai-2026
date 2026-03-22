from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import get_current_year_month_day_utc
from tripletex.my_log import configure_logging

prompts = [
    "Me har oppdaga feil i hovudboka for januar og februar 2026. Gå gjennom alle bilag og finn dei 4 feila: ei postering på feil konto (konto 6500 brukt i staden for 6540, beløp 3450 kr), eit duplikat bilag (konto 6540, beløp 3700 kr), ei manglande MVA-linje (konto 6540, beløp ekskl. 23500 kr manglar MVA på konto 2710), og eit feil beløp (konto 7300, 18600 kr bokført i staden for 11550 kr). Korriger alle feil med rette bilag.",
    "We have discovered errors in the general ledger for January and February 2026. Review all vouchers and find the 4 errors: a posting to the wrong account (account 7100 used instead of 7140, amount 6400 NOK), a duplicate voucher (account 7300, amount 1100 NOK), a missing VAT line (account 6500, amount excl. 19350 NOK missing VAT on account 2710), and an incorrect amount (account 6540, 20500 NOK posted instead of 12150 NOK). Correct all errors with appropriate correction vouchers.",
    "Me har oppdaga feil i hovudboka for januar og februar 2026. Gå gjennom alle bilag og finn dei 4 feila: ei postering på feil konto (konto 6540 brukt i staden for 6860, beløp 4600 kr), eit duplikat bilag (konto 6860, beløp 4150 kr), ei manglande MVA-linje (konto 7000, beløp ekskl. 24100 kr manglar MVA på konto 2710), og eit feil beløp (konto 6300, 21800 kr bokført i staden for 16250 kr). Korriger alle feil med rette bilag.",
    "Hemos descubierto errores en el libro mayor de enero y febrero de 2026. Revise todos los comprobantes y encuentre los 4 errores: un asiento en la cuenta incorrecta (cuenta 6860 usada en lugar de 6590, importe 2200 NOK), un comprobante duplicado (cuenta 6540, importe 2200 NOK), una línea de IVA faltante (cuenta 6590, importe sin IVA 19900 NOK falta IVA en cuenta 2710), y un importe incorrecto (cuenta 6300, 19650 NOK registrado en lugar de 16550 NOK). Corrija todos los errores con asientos correctivos.",
    "We have discovered errors in the general ledger for January and February 2026. Review all vouchers and find the 4 errors: a posting to the wrong account (account 7300 used instead of 7000, amount 1650 NOK), a duplicate voucher (account 6500, amount 3100 NOK), a missing VAT line (account 6540, amount excl. 20950 NOK missing VAT on account 2710), and an incorrect amount (account 6340, 18100 NOK posted instead of 6050 NOK). Correct all errors with appropriate correction vouchers.",
    "Vi har oppdaget feil i hovedboken for januar og februar 2026. Gå gjennom alle bilag og finn de 4 feilene: en postering på feil konto (konto 7140 brukt i stedet for 7100, beløp 5850 kr), et duplisert bilag (konto 7300, beløp 1200 kr), en manglende MVA-linje (konto 6540, beløp ekskl. 13000 kr mangler MVA på konto 2710), og et feil beløp (konto 7100, 19050 kr bokført i stedet for 7100 kr). Korriger alle feil med riktige bilag.",
    "Descobrimos erros no livro razão de janeiro e fevereiro de 2026. Revise todos os vouchers e encontre os 4 erros: um lançamento na conta errada (conta 7300 usada em vez de 7000, valor 7800 NOK), um voucher duplicado (conta 6860, valor 3500 NOK), uma linha de IVA em falta (conta 6500, valor sem IVA 18350 NOK falta IVA na conta 2710), e um valor incorreto (conta 7300, 15000 NOK registado em vez de 10050 NOK). Corrija todos os erros com lançamentos corretivos.",
    "Wir haben Fehler im Hauptbuch für Januar und Februar 2026 entdeckt. Überprüfen Sie alle Belege und finden Sie die 4 Fehler: eine Buchung auf das falsche Konto (Konto 6540 statt 6860, Betrag 3150 NOK), ein doppelter Beleg (Konto 6500, Betrag 1500 NOK), eine fehlende MwSt.-Zeile (Konto 6300, Betrag ohne MwSt. 22350 NOK, fehlende MwSt. auf Konto 2710), und ein falscher Betrag (Konto 6540, 11400 NOK gebucht statt 8550 NOK). Korrigieren Sie alle Fehler mit entsprechenden Korrekturbuchungen.",
    "Descobrimos erros no livro razão de janeiro e fevereiro de 2026. Revise todos os vouchers e encontre os 4 erros: um lançamento na conta errada (conta 7140 usada em vez de 7100, valor 7500 NOK), um voucher duplicado (conta 6540, valor 1000 NOK), uma linha de IVA em falta (conta 4500, valor sem IVA 21500 NOK falta IVA na conta 2710), e um valor incorreto (conta 6860, 17250 NOK registado em vez de 6000 NOK). Corrija todos os erros com lançamentos corretivos.",
    'Wir haben Fehler im Hauptbuch für Januar und Februar 2026 entdeckt. Überprüfen Sie alle Belege und finden Sie die 4 Fehler: eine Buchung auf das falsche Konto (Konto 7300 statt 7000, Betrag 4900 NOK), ein doppelter Beleg (Konto 6540, Betrag 4000 NOK), eine fehlende MwSt.-Zeile (Konto 6500, Betrag ohne MwSt. 23850 NOK, fehlende MwSt. auf Konto 2710), und ein falscher Betrag (Konto 7100, 22300 NOK gebucht statt 19050 NOK). Korrigieren Sie alle Fehler mit entsprechenden Korrekturbuchungen.',
]

logger = configure_logging()


class WrongAccountError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wrong_account: Annotated[int, Field(description="Account number that was incorrectly used")]
    correct_account: Annotated[int, Field(description="Account number that should have been used")]
    amount: float


class DuplicateVoucherError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: Annotated[int, Field(description="Account number of the duplicate posting")]
    amount: float


class MissingVatLineError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: Annotated[int, Field(description="Expense account missing the VAT line")]
    amount_excl_vat: float


class IncorrectAmountError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: Annotated[int, Field(description="Account with the wrong amount")]
    posted_amount: float
    correct_amount: float


class LedgerErrorCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wrong_account_error: WrongAccountError
    duplicate_voucher_error: DuplicateVoucherError
    missing_vat_line_error: MissingVatLineError
    incorrect_amount_error: IncorrectAmountError

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=LedgerErrorCorrection,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Pre-populated vouchers in Jan-Feb 2026 contain exactly 4 errors.
        #   - The duplicate voucher has exactly 2 postings on the same account
        #     for the same amount; we reverse the last one found.
        #   - All referenced account numbers exist in the standard chart of accounts.
        #   - Account 2710 is "Inngående merverdiavgift, høy sats" (25% input VAT).
        #   - Account 1920 is used as the offset (bank) for VAT and amount corrections.
        #   - The missing VAT is always 25% of the excl. amount.
        #
        # Assumptions about what scoring checks:
        #   - The duplicate voucher is reversed (net zero effect on ledger).
        #   - A correction posting moves the wrong-account amount to the right account.
        #   - A VAT posting on account 2710 exists for 25% of the excl. amount.
        #   - The net effect of the amount correction is (posted - correct) removed
        #     from the account.
        #   - Corrections may need to be separate vouchers — if scoring fails,
        #     try splitting the combined voucher into 3 individual ones (3 more writes).
        today = get_current_year_month_day_utc()

        # ── GET ledger accounts ──────────────────────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}
        logger.info("task24.accounts.fetched", extra={"count": len(accts)})

        # ── GET all postings in Jan-Feb 2026 ─────────────────────────────
        postings_r = tripletex_client.get(
            "/ledger/posting",
            params={
                "dateFrom": "2026-01-01",
                "dateTo": "2026-03-01",
                "fields": "id,account(id,number),amount,amountGross,voucher(id),row",
                "count": 1000,
            },
        )
        postings_r.raise_for_status()
        postings = postings_r.json()["values"]
        logger.info("task24.postings.fetched", extra={"count": len(postings)})

        # ── Find and reverse the duplicate voucher ───────────────────────
        dup = self.duplicate_voucher_error
        # Find postings matching the duplicate: same account and amount
        dup_postings = [
            p for p in postings
            if p["account"]["number"] == dup.account and p["amount"] == dup.amount
        ]
        logger.info(
            "task24.duplicate.candidates",
            extra={"count": len(dup_postings), "postings": dup_postings},
        )

        if len(dup_postings) >= 2:
            # Reverse the second (duplicate) voucher
            dup_voucher_id = dup_postings[-1]["voucher"]["id"]
        else:
            # Fallback: reverse the only match
            dup_voucher_id = dup_postings[0]["voucher"]["id"]

        logger.info(
            "task24.duplicate.reversing",
            extra={"voucher_id": dup_voucher_id},
        )
        reverse_r = tripletex_client.put(
            f"/ledger/voucher/{dup_voucher_id}/:reverse",
            params={"date": today},
        )
        logger.info(
            "task24.duplicate.response",
            extra={"status": reverse_r.status_code, "body": reverse_r.json()},
        )
        reverse_r.raise_for_status()

        # ── Build combined correction voucher ────────────────────────────
        correction_postings = []
        row = 1

        # 1. Wrong account: credit wrong account, debit correct account
        wa = self.wrong_account_error
        correction_postings.append({
            "account": {"id": accts[wa.wrong_account]["id"]},
            "amountCurrency": -wa.amount,
            "amountGross": -wa.amount,
            "amountGrossCurrency": -wa.amount,
            "row": row,
        })
        row += 1
        correction_postings.append({
            "account": {"id": accts[wa.correct_account]["id"]},
            "amountCurrency": wa.amount,
            "amountGross": wa.amount,
            "amountGrossCurrency": wa.amount,
            "row": row,
        })
        row += 1

        # 2. Missing VAT line: debit 2710 (inngående MVA), credit 1920 (bank)
        vat = self.missing_vat_line_error
        vat_amount = vat.amount_excl_vat * 0.25
        correction_postings.append({
            "account": {"id": accts[2710]["id"]},
            "amountCurrency": vat_amount,
            "amountGross": vat_amount,
            "amountGrossCurrency": vat_amount,
            "row": row,
        })
        row += 1
        correction_postings.append({
            "account": {"id": accts[1920]["id"]},
            "amountCurrency": -vat_amount,
            "amountGross": -vat_amount,
            "amountGrossCurrency": -vat_amount,
            "row": row,
        })
        row += 1

        # 3. Incorrect amount: credit the excess (or debit the shortfall)
        ia = self.incorrect_amount_error
        diff = ia.posted_amount - ia.correct_amount  # positive = overstated
        correction_postings.append({
            "account": {"id": accts[ia.account]["id"]},
            "amountCurrency": -diff,
            "amountGross": -diff,
            "amountGrossCurrency": -diff,
            "row": row,
        })
        row += 1
        correction_postings.append({
            "account": {"id": accts[1920]["id"]},
            "amountCurrency": diff,
            "amountGross": diff,
            "amountGrossCurrency": diff,
            "row": row,
        })
        row += 1

        voucher_payload = {
            "date": today,
            "description": "Korreksjon av feil i hovedbok",
            "postings": correction_postings,
        }
        logger.info("task24.correction.creating", extra={"payload": voucher_payload})
        correction_r = tripletex_client.post("/ledger/voucher", json=voucher_payload)
        logger.info(
            "task24.correction.response",
            extra={"status": correction_r.status_code, "body": correction_r.json()},
        )
        correction_r.raise_for_status()

        logger.info("task24.completed")


if __name__ == "__main__":
    for prompt in prompts:
        res = LedgerErrorCorrection.parse(prompt)
        print(f"{prompt[:80]}...")
        print(f"  {res=}")
        print()
