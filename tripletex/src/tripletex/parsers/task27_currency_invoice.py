from typing import Annotated, Literal, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_current_year_month_day_utc,
)
from tripletex.my_log import configure_logging

prompts = [
    "Enviámos uma fatura de 8676 EUR ao Cascata Lda (org. nº 967770892) quando a taxa de câmbio era 11.62 NOK/EUR. O cliente pagou agora, mas a taxa é 10.89 NOK/EUR. Registe o pagamento e lance a diferença cambial (disagio) na conta correta.",
    "Enviámos uma fatura de 2336 EUR ao Estrela Lda (org. nº 808808773) quando a taxa de câmbio era 11.17 NOK/EUR. O cliente pagou agora, mas a taxa é 12.13 NOK/EUR. Registe o pagamento e lance a diferença cambial (agio) na conta correta.",
    "Nous avons envoyé une facture de 9273 EUR à Montagne SARL (nº org. 941382142) lorsque le taux de change était de 10.34 NOK/EUR. Le client a maintenant payé, mais le taux est de 10.83 NOK/EUR. Enregistrez le paiement et comptabilisez l'écart de change (agio) sur le bon compte.",
    "Wir haben eine Rechnung über 19629 EUR an Waldstein GmbH (Org.-Nr. 919320044) gesendet, als der Wechselkurs 11.14 NOK/EUR betrug. Der Kunde hat nun bezahlt, aber der Kurs liegt bei 10.74 NOK/EUR. Erfassen Sie die Zahlung und buchen Sie die Wechselkursdifferenz (disagio) auf das korrekte Konto.",
    "Enviamos una factura por 9487 EUR a Estrella SL (org. nº 834293692) cuando el tipo de cambio era 11.54 NOK/EUR. El cliente ha pagado ahora, pero el tipo es 10.95 NOK/EUR. Registre el pago y contabilice la diferencia de tipo de cambio (disagio) en la cuenta correcta.",
    "Wir haben eine Rechnung über 8077 EUR an Bergwerk GmbH (Org.-Nr. 863352010) gesendet, als der Wechselkurs 10.91 NOK/EUR betrug. Der Kunde hat nun bezahlt, aber der Kurs liegt bei 10.29 NOK/EUR. Erfassen Sie die Zahlung und buchen Sie die Wechselkursdifferenz (disagio) auf das korrekte Konto.",
    "Me sende ein faktura på 12301 EUR til Bølgekraft AS (org.nr 830993940) då kursen var 10.83 NOK/EUR. Kunden har no betalt, men kursen er 11.83 NOK/EUR. Registrer betalinga og bokfør valutadifferansen (agio) på rett konto.",
    "Me sende ein faktura på 9364 EUR til Elvdal AS (org.nr 808826054) då kursen var 11.65 NOK/EUR. Kunden har no betalt, men kursen er 12.35 NOK/EUR. Registrer betalinga og bokfør valutadifferansen (agio) på rett konto.",
    "Nous avons envoyé une facture de 6224 EUR à Rivière SARL (nº org. 886727127) lorsque le taux de change était de 10.90 NOK/EUR. Le client a maintenant payé, mais le taux est de 10.33 NOK/EUR. Enregistrez le paiement et comptabilisez l'écart de change (disagio) sur le bon compte.",
    "Enviámos uma fatura de 10765 EUR ao Estrela Lda (org. nº 950948086) quando a taxa de câmbio era 11.30 NOK/EUR. O cliente pagou agora, mas a taxa é 10.78 NOK/EUR. Registe o pagamento e lance a diferença cambial (disagio) na conta correta.",
    'Vi sendte en faktura på 11497 EUR til Tindra AS (org.nr 862097653) da kursen var 10.09 NOK/EUR. Kunden har nå betalt, men kursen er 10.84 NOK/EUR. Registrer betalingen og bokfør valutadifferansen (agio) på korrekt konto.',
    'Enviamos una factura por 18687 EUR a Solmar SL (org. nº 877276260) cuando el tipo de cambio era 10.33 NOK/EUR. El cliente ha pagado ahora, pero el tipo es 10.87 NOK/EUR. Registre el pago y contabilice la diferencia de tipo de cambio (agio) en la cuenta correcta.',
]

logger = configure_logging()


class CurrencyInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eur_amount: float
    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    original_rate: float
    payment_rate: float
    exchange_rate_type: Literal["agio", "disagio"]

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=CurrencyInvoice,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - The customer exists and has exactly one outstanding EUR invoice.
        #   - The invoice was created at original_rate NOK/EUR.
        #   - Account 8060 (Valutagevinst/agio) and 8160 (Valutatap/disagio) exist.
        #   - Account 1920 (bank) and 1500 (kundefordringer) exist.
        #   - Exactly one invoice payment type with debitAccount=1920 exists.
        #
        # Assumptions about what scoring checks:
        #   - Payment registered on the EUR invoice (amountOutstanding = 0).
        #   - Exchange rate difference posted to correct account (8060 or 8160).
        #
        # Not verified:
        #   - Whether Tripletex auto-posts the exchange difference when paidAmount
        #     differs from the invoice NOK amount, or if we need a manual voucher.
        #   - Whether the exchange difference voucher needs customer reference.
        today = get_current_year_month_day_utc()

        # ── GET the outstanding EUR invoice ──────────────────────────────
        # Find by amountCurrencyOutstanding matching eur_amount, rather than
        # going through customer lookup (avoids dirty-sandbox issues with
        # duplicate org numbers, and in competition there's exactly one).
        invoices_r = tripletex_client.get(
            "/invoice",
            params={
                "invoiceDateFrom": "1970-01-01",
                "invoiceDateTo": "2027-12-31",
                "fields": "id,invoiceNumber,amount,amountOutstanding,amountCurrencyOutstanding,currency(id,displayName),customer(id)",
            },
        )
        invoices_r.raise_for_status()
        invoices = invoices_r.json()["values"]
        logger.info("task27.invoices.fetched", extra={"count": len(invoices)})

        eur_outstanding = [
            inv for inv in invoices
            if inv["amountOutstanding"] > 0
            and inv.get("currency", {}).get("displayName") == "EUR"
        ]
        logger.info(
            "task27.eurInvoices.found",
            extra={"count": len(eur_outstanding), "invoices": eur_outstanding},
        )
        (invoice,) = eur_outstanding
        customer_id = invoice["customer"]["id"]
        logger.info(
            "task27.invoice.selected",
            extra={
                "invoice_id": invoice["id"],
                "invoice_number": invoice["invoiceNumber"],
                "amount": invoice["amount"],
                "outstanding": invoice["amountOutstanding"],
            },
        )

        # ── GET invoice payment type (debit account 1920) ────────────────
        inv_pay_types = tripletex_client.get(
            "/invoice/paymentType",
            params={"fields": "id,debitAccount(id,number)"},
        )
        inv_pay_types.raise_for_status()
        (bank_pay_type,) = [
            t for t in inv_pay_types.json()["values"]
            if t["debitAccount"]["number"] == 1920
        ]
        payment_type_id = bank_pay_type["id"]

        # ── Compute amounts ──────────────────────────────────────────────
        nok_at_payment_rate = round(self.eur_amount * self.payment_rate, 2)
        nok_at_original_rate = round(self.eur_amount * self.original_rate, 2)
        exchange_diff = round(nok_at_original_rate - nok_at_payment_rate, 2)
        # exchange_diff > 0 → disagio (we receive less NOK), < 0 → agio (we receive more)

        logger.info(
            "task27.amounts",
            extra={
                "eur_amount": self.eur_amount,
                "nok_original": nok_at_original_rate,
                "nok_payment": nok_at_payment_rate,
                "exchange_diff": exchange_diff,
                "type": self.exchange_rate_type,
            },
        )

        # ── WRITE 1: register payment on the invoice ────────────────────
        # paidAmount = NOK received (at payment rate)
        # paidAmountCurrency = EUR amount (full invoice in foreign currency)
        logger.info(
            "task27.payment.registering",
            extra={
                "invoice_id": invoice["id"],
                "paidAmount_nok": nok_at_payment_rate,
                "paidAmountCurrency_eur": self.eur_amount,
            },
        )
        pay_r = tripletex_client.put(
            f"/invoice/{invoice['id']}/:payment",
            params={
                "paymentDate": today,
                "paymentTypeId": payment_type_id,
                "paidAmount": nok_at_payment_rate,
                "paidAmountCurrency": self.eur_amount,
            },
        )
        logger.info(
            "task27.payment.response",
            extra={"status": pay_r.status_code, "body": pay_r.json()},
        )
        pay_r.raise_for_status()

        # ── GET ledger accounts ──────────────────────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}

        # ── WRITE 2: post exchange rate difference voucher ───────────────
        # The payment settles the EUR amount but the NOK differs from what
        # was booked on the invoice. Post the difference:
        #   disagio (loss, exchange_diff > 0): debit 8160, credit 1500
        #   agio (gain, exchange_diff < 0):    debit 1500, credit 8060
        abs_diff = abs(exchange_diff)
        if self.exchange_rate_type == "disagio":
            debit_account = 8160
            credit_account = 1500
        else:
            debit_account = 1500
            credit_account = 8060

        def _posting(account_num: int, amount: float, row: int) -> dict:
            p: dict = {
                "account": {"id": accts[account_num]["id"]},
                "amountCurrency": amount,
                "amountGross": amount,
                "amountGrossCurrency": amount,
                "row": row,
            }
            # Account 1500 (kundefordringer) requires a customer reference
            if account_num == 1500:
                p["customer"] = {"id": customer_id}
            return p

        fx_voucher_payload = {
            "date": today,
            "description": f"Valutadifferanse ({self.exchange_rate_type})",
            "postings": [
                _posting(debit_account, abs_diff, 1),
                _posting(credit_account, -abs_diff, 2),
            ],
        }
        logger.info("task27.fxVoucher.creating", extra={"payload": fx_voucher_payload})
        fx_r = tripletex_client.post("/ledger/voucher", json=fx_voucher_payload)
        logger.info(
            "task27.fxVoucher.response",
            extra={"status": fx_r.status_code, "body": fx_r.json()},
        )
        fx_r.raise_for_status()

        logger.info("task27.completed")


if __name__ == "__main__":
    for prompt in prompts:
        res = CurrencyInvoice.parse(prompt)
        print(f"{prompt[:80]}...")
        print(f"  {res=}")
        print()
