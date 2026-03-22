from typing import Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import get_current_year_month_day_utc
from tripletex.my_log import configure_logging

prompts = [
    "Einer Ihrer Kunden hat eine uberfallige Rechnung. Finden Sie die uberfallige Rechnung und buchen Sie eine Mahngebuhr von 70 NOK. Soll Forderungen (1500), Haben Mahngebuhren (3400). Erstellen Sie außerdem eine Rechnung über die Mahngebühr an den Kunden und senden Sie diese. Registrieren Sie zusätzlich eine Teilzahlung von 5000 NOK auf der überfälligen Rechnung.",
    "Ein av kundane dine har ein forfallen faktura. Finn den forfalne fakturaen og bokfor eit purregebyr pa 35 kr. Debet kundefordringar (1500), kredit purregebyr (3400). Opprett også ein faktura for purregebyret til kunden og send den. Registrer i tillegg ei delbetaling på 5000 kr på den forfalne fakturaen.",
    "Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 60 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.",
    "Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 35 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.",
    "En av kundene dine har en forfalt faktura. Finn den forfalte fakturaen og bokfor et purregebyr pa 40 kr. Debet kundefordringer (1500), kredit purregebyr (3400). Opprett også en faktura for purregebyret til kunden og send den. Registrer i tillegg en delbetaling på 5000 kr på den forfalte fakturaen.",
    "L'un de vos clients a une facture en retard. Trouvez la facture en retard et enregistrez des frais de rappel de 70 NOK. Debit creances clients (1500), credit revenus de rappel (3400). Créez également une facture pour les frais de rappel au client et envoyez-la. De plus, enregistrez un paiement partiel de 5000 NOK sur la facture en retard.",
    "One of your customers has an overdue invoice. Find the overdue invoice and post a reminder fee of 70 NOK. Debit accounts receivable (1500), credit reminder fees (3400). Also create an invoice for the reminder fee to the customer and send it. Additionally, register a partial payment of 5000 NOK on the overdue invoice.",
    "Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 65 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.",
    "Uno de sus clientes tiene una factura vencida. Encuentre la factura vencida y registre un cargo por recordatorio de 55 NOK. Debito cuentas por cobrar (1500), credito ingresos por recordatorio (3400). También cree una factura por la tarifa de recordatorio al cliente y envíela. Además, registre un pago parcial de 5000 NOK en la factura vencida.",
    "Um dos seus clientes tem uma fatura vencida. Encontre a fatura vencida e registe uma taxa de lembrete de 40 NOK. Debito contas a receber (1500), credito receitas de lembrete (3400). Também crie uma fatura para a taxa de lembrete ao cliente e envie-a. Além disso, registe um pagamento parcial de 5000 NOK na fatura vencida.",
    'L\'un de vos clients a une facture en retard. Trouvez la facture en retard et enregistrez des frais de rappel de 50 NOK. Debit creances clients (1500), credit revenus de rappel (3400). Créez également une facture pour les frais de rappel au client et envoyez-la. De plus, enregistrez un paiement partiel de 5000 NOK sur la facture en retard.',
    'En av kundene dine har en forfalt faktura. Finn den forfalte fakturaen og bokfor et purregebyr pa 50 kr. Debet kundefordringer (1500), kredit purregebyr (3400). Opprett også en faktura for purregebyret til kunden og send den. Registrer i tillegg en delbetaling på 5000 kr på den forfalte fakturaen.',
]

logger = configure_logging()


class OverdueReminder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reminder_fee: float
    partial_payment_amount: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=OverdueReminder,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Exactly one customer invoice is overdue (invoiceDueDate < today,
        #     amountOutstanding > 0).
        #   - Accounts 1500 (Kundefordringer) and 3400 exist in chart of accounts.
        #   - Account 1920 (Bankinnskudd) exists for bank payment.
        #   - Exactly one invoice payment type with debitAccount=1920 exists.
        #   - The overdue invoice has amountOutstanding >= partial_payment_amount (5000).
        #
        # Assumptions about what scoring checks:
        #   - A voucher exists with debit 1500 / credit 3400 for the reminder fee.
        #   - A reminder invoice exists for the fee amount, sent to the customer.
        #   - The overdue invoice shows reduced amountOutstanding after partial payment.
        #
        # Not verified:
        #   - Whether the reminder invoice needs a specific VAT treatment (we use 0%).
        #   - Whether the voucher description matters.
        #   - Whether the reminder invoice description matters.
        today = get_current_year_month_day_utc()

        # ── GET all invoices, find the overdue one ───────────────────────
        invoices_r = tripletex_client.get(
            "/invoice",
            params={
                "invoiceDateFrom": "1970-01-01",
                "invoiceDateTo": "2027-12-31",
                "fields": "id,invoiceNumber,invoiceDueDate,amount,amountOutstanding,customer(id,name)",
            },
        )
        invoices_r.raise_for_status()
        invoices = invoices_r.json()["values"]
        logger.info("task25.invoices.fetched", extra={"count": len(invoices)})

        overdue = [
            inv for inv in invoices
            if inv["amountOutstanding"] > 0 and inv["invoiceDueDate"] < today
        ]
        logger.info(
            "task25.overdue.found",
            extra={"count": len(overdue), "overdue": overdue},
        )
        (overdue_invoice,) = overdue
        customer_id = overdue_invoice["customer"]["id"]
        logger.info(
            "task25.overdue.selected",
            extra={
                "invoice_id": overdue_invoice["id"],
                "invoice_number": overdue_invoice["invoiceNumber"],
                "customer": overdue_invoice["customer"]["name"],
                "outstanding": overdue_invoice["amountOutstanding"],
            },
        )

        # ── GET ledger accounts ──────────────────────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}

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

        # ── WRITE 1: POST reminder fee voucher (debit 1500, credit 3400) ─
        fee_voucher_payload = {
            "date": today,
            "description": "Purregebyr",
            "postings": [
                {
                    "account": {"id": accts[1500]["id"]},
                    "amountCurrency": self.reminder_fee,
                    "amountGross": self.reminder_fee,
                    "amountGrossCurrency": self.reminder_fee,
                    "customer": {"id": customer_id},
                    "row": 1,
                },
                {
                    "account": {"id": accts[3400]["id"]},
                    "amountCurrency": -self.reminder_fee,
                    "amountGross": -self.reminder_fee,
                    "amountGrossCurrency": -self.reminder_fee,
                    "row": 2,
                },
            ],
        }
        logger.info("task25.feeVoucher.creating", extra={"payload": fee_voucher_payload})
        fee_r = tripletex_client.post("/ledger/voucher", json=fee_voucher_payload)
        logger.info(
            "task25.feeVoucher.response",
            extra={"status": fee_r.status_code, "body": fee_r.json()},
        )
        fee_r.raise_for_status()

        # ── WRITE 2: POST reminder invoice to customer ───────────────────
        invoice_payload = {
            "invoiceDate": today,
            "invoiceDueDate": get_current_year_month_day_utc(days_offset_forward=14),
            "customer": {"id": customer_id},
            "orders": [
                {
                    "customer": {"id": customer_id},
                    "orderDate": today,
                    "deliveryDate": today,
                    "orderLines": [
                        {
                            "count": 1,
                            "unitPriceExcludingVatCurrency": self.reminder_fee,
                            "vatType": {"id": 0},
                            "description": "Purregebyr",
                        }
                    ],
                }
            ],
        }
        logger.info("task25.reminderInvoice.creating", extra={"payload": invoice_payload})
        inv_r = tripletex_client.post(
            "/invoice",
            params={"sendToCustomer": True},
            json=invoice_payload,
        )
        logger.info(
            "task25.reminderInvoice.response",
            extra={"status": inv_r.status_code, "body": inv_r.json()},
        )
        inv_r.raise_for_status()

        # ── WRITE 3: PUT partial payment on overdue invoice ──────────────
        logger.info(
            "task25.partialPayment.registering",
            extra={
                "invoice_id": overdue_invoice["id"],
                "amount": self.partial_payment_amount,
            },
        )
        pay_r = tripletex_client.put(
            f"/invoice/{overdue_invoice['id']}/:payment",
            params={
                "paymentDate": today,
                "paymentTypeId": payment_type_id,
                "paidAmount": self.partial_payment_amount,
                "paidAmountCurrency": self.partial_payment_amount,
            },
        )
        logger.info(
            "task25.partialPayment.response",
            extra={"status": pay_r.status_code, "body": pay_r.json()},
        )
        pay_r.raise_for_status()

        logger.info("task25.completed")


if __name__ == "__main__":
    for prompt in prompts:
        res = OverdueReminder.parse(prompt)
        print(f"{prompt[:80]}...")
        print(f"  {res=}")
        print()
