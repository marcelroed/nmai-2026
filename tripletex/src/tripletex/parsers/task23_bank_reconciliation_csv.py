from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.my_log import configure_logging

prompts = [
    "Reconcilie o extrato bancario (CSV anexo) com as faturas em aberto no Tripletex. Relacione os pagamentos recebidos com as faturas de clientes e os pagamentos efetuados com as faturas de fornecedores. Trate os pagamentos parciais corretamente.",
    "Avstem bankutskrifta (vedlagt CSV) mot opne fakturaer i Tripletex. Match innbetalingar til kundefakturaer og utbetalingar til leverandorfakturaer. Handter delbetalingar korrekt.",
    "Gleichen Sie den Kontoauszug (beigefuegte CSV) mit den offenen Rechnungen in Tripletex ab. Ordnen Sie eingehende Zahlungen Kundenrechnungen und ausgehende Zahlungen Lieferantenrechnungen zu. Behandeln Sie Teilzahlungen korrekt.",
    "Concilia el extracto bancario (CSV adjunto) con las facturas abiertas en Tripletex. Relaciona los pagos entrantes con las facturas de clientes y los pagos salientes con las facturas de proveedores. Maneja los pagos parciales correctamente.",
    "Rapprochez le releve bancaire (CSV ci-joint) avec les factures ouvertes dans Tripletex. Associez les paiements entrants aux factures clients et les paiements sortants aux factures fournisseurs. Gerez correctement les paiements partiels.",
    "Reconcile the bank statement (attached CSV) against open invoices in Tripletex. Match incoming payments to customer invoices and outgoing payments to supplier invoices. Handle partial payments correctly.",
    'Avstem bankutskriften (vedlagt CSV) mot apne fakturaer i Tripletex. Match innbetalinger til kundefakturaer og utbetalinger til leverandorfakturaer. Handter delbetalinger korrekt.',
]

attachments = [
    "data/files/raw/bankutskrift_en_05.csv",
    "data/files/raw/bankutskrift_en_06.csv",
    "data/files/raw/bankutskrift_de_08.csv",
    "data/files/raw/bankutskrift_es_08.csv",
    "data/files/raw/bankutskrift_fr_03.csv",
    "data/files/raw/bankutskrift_pt_04.csv",
]

logger = configure_logging()


class IncomingPayment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    customer_name: str
    invoice_number: int
    amount: float


class OutgoingPayment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    supplier_name: str
    amount: Annotated[
        float,
        Field(description="Positive amount (absolute value of the outgoing payment)"),
    ]


class BankFee(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    amount: Annotated[float, Field(description="Positive amount of the fee")]


class Interest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    amount: float
    is_income: Annotated[
        bool,
        Field(
            description="True if interest received (positive), False if interest paid (negative)"
        ),
    ]


class BankReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incoming_payments: list[IncomingPayment]
    outgoing_payments: list[OutgoingPayment]
    bank_fees: list[BankFee]
    interest: list[Interest]

    @classmethod
    def parse(cls, prompt: str, attachment: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n--- ATTACHMENT ---\n{attachment}",
                }
            ],
            output_format=BankReconciliation,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # TODO: Not fully battle-tested — moving on to other tasks.
        #
        # Assumptions about the clean competition environment:
        #   - Customer invoices exist with invoiceNumber matching the CSV (e.g. 1001).
        #   - Supplier invoices exist, one per supplier per CSV payment line.
        #   - Exactly one invoice payment type with debitAccount=1920 exists.
        #   - At least one outgoing payment type with creditAccount=1920 exists.
        #   - Standard chart of accounts: 7770 (bank fees), 8040 (interest income),
        #     8150 (interest expense), 1920 (bank).
        #
        # Assumptions about what scoring checks:
        #   - Each customer invoice shows reduced amountOutstanding after payment.
        #   - Each supplier invoice has a registered payment.
        #   - A voucher exists for bank fees with correct debit/credit amounts.
        #   - A voucher exists for interest with correct debit/credit amounts.
        #
        # Known gaps:
        #   - Outgoing payment matching by supplier name + amount is heuristic;
        #     may mis-match if a supplier has multiple invoices with the same amount.
        #   - Partial payments not yet verified against real competition data.
        #   - Bank fee / interest voucher accounts (7770, 8040, 8150) assumed correct
        #     but not confirmed by scoring.
        # ── GET invoice payment type (match by debit account 1920) ───────
        inv_pay_types = tripletex_client.get(
            "/invoice/paymentType",
            params={"fields": "id,description,debitAccount(id,number)"},
        )
        inv_pay_types.raise_for_status()
        inv_pay_types_data = inv_pay_types.json()["values"]
        logger.info(
            "task23.invoicePaymentTypes.fetched",
            extra={"payment_types": inv_pay_types_data},
        )
        (bank_pay_type,) = [
            t for t in inv_pay_types_data if t["debitAccount"]["number"] == 1920
        ]
        inv_payment_type_id = bank_pay_type["id"]

        # ── GET outgoing payment type (first with credit account 1920) ───
        out_pay_types = tripletex_client.get(
            "/ledger/paymentTypeOut",
            params={"fields": "id,description,creditAccount(id,number)"},
        )
        out_pay_types.raise_for_status()
        out_pay_types_data = out_pay_types.json()["values"]
        logger.info(
            "task23.outgoingPaymentTypes.fetched",
            extra={"payment_types": out_pay_types_data},
        )
        # Multiple types may use account 1920; pick the first (standard bank payment)
        out_bank_pay_type = next(
            t for t in out_pay_types_data if t["creditAccount"]["number"] == 1920
        )
        out_payment_type_id = out_bank_pay_type["id"]

        # ── GET all open invoices ────────────────────────────────────────
        invoices_r = tripletex_client.get(
            "/invoice",
            params={
                "invoiceDateFrom": "1970-01-01",
                "invoiceDateTo": "2027-12-31",
                "fields": "id,invoiceNumber,amount,amountOutstanding",
            },
        )
        invoices_r.raise_for_status()
        invoices = invoices_r.json()["values"]
        invoice_by_number = {inv["invoiceNumber"]: inv for inv in invoices}
        logger.info(
            "task23.invoices.fetched",
            extra={"count": len(invoices), "invoices": invoices},
        )

        # ── GET all supplier invoices ────────────────────────────────────
        supplier_invoices_r = tripletex_client.get(
            "/supplierInvoice",
            params={
                "invoiceDateFrom": "1970-01-01",
                "invoiceDateTo": "2027-12-31",
                "fields": "id,invoiceNumber,supplier(id,name),amount",
            },
        )
        supplier_invoices_r.raise_for_status()
        supplier_invoices = supplier_invoices_r.json()["values"]
        logger.info(
            "task23.supplierInvoices.fetched",
            extra={
                "count": len(supplier_invoices),
                "supplier_invoices": supplier_invoices,
            },
        )

        # ── WRITE: register incoming payments (customer invoices) ────────
        for payment in self.incoming_payments:
            invoice = invoice_by_number.get(payment.invoice_number)
            if not invoice:
                logger.error(
                    "task23.incoming.invoiceNotFound",
                    extra={"invoice_number": payment.invoice_number},
                )
                continue

            logger.info(
                "task23.incoming.registering",
                extra={
                    "invoice_id": invoice["id"],
                    "invoice_number": payment.invoice_number,
                    "amount": payment.amount,
                    "date": payment.date,
                },
            )
            r = tripletex_client.put(
                f"/invoice/{invoice['id']}/:payment",
                params={
                    "paymentDate": payment.date,
                    "paymentTypeId": inv_payment_type_id,
                    "paidAmount": payment.amount,
                    "paidAmountCurrency": payment.amount,
                },
            )
            logger.info(
                "task23.incoming.response",
                extra={"status": r.status_code, "body": r.json()},
            )
            r.raise_for_status()

        # ── WRITE: register outgoing payments (supplier invoices) ────────
        used_supplier_invoice_ids: set[int] = set()
        for payment in self.outgoing_payments:
            # Match supplier invoice by name, prefer amount match, skip already-used
            matching = [
                si
                for si in supplier_invoices
                if si["supplier"]["name"].lower() == payment.supplier_name.lower()
                and si["id"] not in used_supplier_invoice_ids
            ]
            if not matching:
                logger.error(
                    "task23.outgoing.supplierNotFound",
                    extra={"supplier_name": payment.supplier_name},
                )
                continue

            # Try exact amount match first, fall back to first available
            si = next(
                (s for s in matching if s["amount"] == payment.amount),
                matching[0],
            )
            used_supplier_invoice_ids.add(si["id"])
            logger.info(
                "task23.outgoing.registering",
                extra={
                    "supplier_invoice_id": si["id"],
                    "supplier_name": payment.supplier_name,
                    "amount": payment.amount,
                    "date": payment.date,
                },
            )
            r = tripletex_client.post(
                f"/supplierInvoice/{si['id']}/:addPayment",
                params={
                    "paymentDate": payment.date,
                    "paymentTypeId": out_payment_type_id,
                    "paidAmount": payment.amount,
                    "paidAmountCurrency": payment.amount,
                },
            )
            logger.info(
                "task23.outgoing.response",
                extra={"status": r.status_code, "body": r.json()},
            )
            r.raise_for_status()

        # ── WRITE: post bank fees + interest as a single voucher ─────────
        if self.bank_fees or self.interest:
            # GET ledger accounts
            accounts_r = tripletex_client.get(
                "/ledger/account",
                params={"fields": "id,number", "count": 1000},
            )
            accounts_r.raise_for_status()
            accts = {a["number"]: a for a in accounts_r.json()["values"]}

            postings = []
            row = 1
            for fee in self.bank_fees:
                # Debit 7770 (Bank og kortgebyrer), Credit 1920 (Bank)
                postings.append(
                    {
                        "account": {"id": accts[7770]["id"]},
                        "amountCurrency": fee.amount,
                        "amountGross": fee.amount,
                        "amountGrossCurrency": fee.amount,
                        "row": row,
                    }
                )
                row += 1
                postings.append(
                    {
                        "account": {"id": accts[1920]["id"]},
                        "amountCurrency": -fee.amount,
                        "amountGross": -fee.amount,
                        "amountGrossCurrency": -fee.amount,
                        "row": row,
                    }
                )
                row += 1

            for interest in self.interest:
                if interest.is_income:
                    # Interest income: Debit 1920 (Bank), Credit 8040 (Renteinntekt)
                    postings.append(
                        {
                            "account": {"id": accts[1920]["id"]},
                            "amountCurrency": interest.amount,
                            "amountGross": interest.amount,
                            "amountGrossCurrency": interest.amount,
                            "row": row,
                        }
                    )
                    row += 1
                    postings.append(
                        {
                            "account": {"id": accts[8040]["id"]},
                            "amountCurrency": -interest.amount,
                            "amountGross": -interest.amount,
                            "amountGrossCurrency": -interest.amount,
                            "row": row,
                        }
                    )
                    row += 1
                else:
                    # Interest expense: Debit 8150 (Rentekostnad), Credit 1920 (Bank)
                    postings.append(
                        {
                            "account": {"id": accts[8150]["id"]},
                            "amountCurrency": interest.amount,
                            "amountGross": interest.amount,
                            "amountGrossCurrency": interest.amount,
                            "row": row,
                        }
                    )
                    row += 1
                    postings.append(
                        {
                            "account": {"id": accts[1920]["id"]},
                            "amountCurrency": -interest.amount,
                            "amountGross": -interest.amount,
                            "amountGrossCurrency": -interest.amount,
                            "row": row,
                        }
                    )
                    row += 1

            if postings:
                # Use the date of the first fee/interest entry
                voucher_date = (
                    self.bank_fees[0].date if self.bank_fees else self.interest[0].date
                )
                voucher_payload = {
                    "date": voucher_date,
                    "description": "Bankgebyr og renter",
                    "postings": postings,
                }
                logger.info(
                    "task23.feeInterestVoucher.creating",
                    extra={"payload": voucher_payload},
                )
                r = tripletex_client.post("/ledger/voucher", json=voucher_payload)
                logger.info(
                    "task23.feeInterestVoucher.response",
                    extra={"status": r.status_code, "body": r.json()},
                )
                r.raise_for_status()

        logger.info("task23.completed")


if __name__ == "__main__":
    from pathlib import Path

    from tripletex.herman_tasks.utils import TripletexCredentials

    tripletex_client = TripletexCredentials.placeholder_TODO().to_client()

    prompt = prompts[5]  # English
    attachment = Path(attachments[0]).read_text()

    parsed = BankReconciliation.parse(prompt, attachment)
    logger.info("task23.parsed", extra={"parsed": parsed.model_dump()})

    parsed.solve(tripletex_client=tripletex_client)
