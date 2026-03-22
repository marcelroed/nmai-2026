from datetime import date, timedelta
from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "Voce recebeu uma fatura de fornecedor (ver PDF anexo). Registe a fatura no Tripletex. Crie o fornecedor se nao existir. Use a conta de despesas correta e o IVA de entrada.",
    "Du har mottatt en leverandorfaktura (se vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandoren hvis den ikke finnes. Bruk riktig utgiftskonto og inngaende MVA.",
    "Du har motteke ein leverandorfaktura (sjaa vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandoren viss den ikkje finst. Bruk rett utgiftskonto og inngaaande MVA.",
    "Vous avez recu une facture fournisseur (voir PDF ci-joint). Enregistrez la facture dans Tripletex. Creez le fournisseur s'il n'existe pas. Utilisez le bon compte de charges et la TVA deductible.",
]

attachments = [
    "data/files/parsed/leverandorfaktura_nb_05.txt",
    "data/files/parsed/leverandorfaktura_nb_08.txt",
    "data/files/parsed/leverandorfaktura_fr_03.txt",
    "data/files/parsed/leverandorfaktura_fr_08.txt",
]


class SupplierInvoiceFromPDF(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    supplier_street_address: str
    supplier_postal_code: str
    supplier_city: str
    supplier_bank_account: str
    invoice_number: Annotated[str, Field(pattern=r"^INV-\d{4}-\d+$")]
    invoice_year: int
    invoice_month: int
    invoice_day: int
    due_year: int
    due_month: int
    due_day: int
    description: str
    amount_excl_vat: float
    vat_rate: float
    amount_incl_vat: float
    expense_account: Annotated[str, Field(pattern=r"^\d{4}$")]

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
            output_format=SupplierInvoiceFromPDF,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        invoice_date = date(
            self.invoice_year, self.invoice_month, self.invoice_day
        ).isoformat()
        due_date = date(self.due_year, self.due_month, self.due_day).isoformat()

        supplier_payload = {
            "name": self.supplier_name,
            "organizationNumber": self.org_number,
            "postalAddress": {
                "addressLine1": self.supplier_street_address,
                "postalCode": self.supplier_postal_code,
                "city": self.supplier_city,
            },
        }
        supplier_post_r = tripletex_client.post("/supplier", json=supplier_payload)
        supplier_post_r.raise_for_status()
        supplier = supplier_post_r.json()["value"]
        supplier

        ledger_accounts = tripletex_client.get(
            "/ledger/account",
            params={
                "number": str(self.expense_account),
                "isApplicableForSupplierInvoice": "true",
                "fields": "id,number,name,ledgerType,isApplicableForSupplierInvoice,vatLocked,vatType(id,number),legalVatTypes(id,number)",
                "count": 1000,
            },
        )
        ledger_accounts.raise_for_status()
        ledger_accounts = ledger_accounts.json()["values"]

        (expense_account,) = [
            x for x in ledger_accounts if str(x["number"]) == str(self.expense_account)
        ]

        if not expense_account["isApplicableForSupplierInvoice"]:
            raise ValueError(
                f"Expense account {self.expense_account} is not applicable for supplier invoices"
            )

        expense_account

        # TODO: check that this should actually be
        incoming_vat_types = tripletex_client.get(
            "/ledger/vatType",
            params={
                "number": 1,
            },
        )
        incoming_vat_types.raise_for_status()
        incoming_vat_types = incoming_vat_types.json()["values"]

        (incoming_vat_type,) = [x for x in incoming_vat_types if x["number"] == 1]

        vat_type = (
            expense_account["vatType"]
            if expense_account["vatLocked"]
            else incoming_vat_type
        )

        legal_vat_type_ids = {x["id"] for x in expense_account.get("legalVatTypes", [])}
        if legal_vat_type_ids and vat_type["id"] not in legal_vat_type_ids:
            raise ValueError(
                f"VAT type {vat_type['id']} is not legal for expense account {expense_account['number']}"
            )

        vat_type

        supplier_ledger_account = supplier.get("ledgerAccount")
        if supplier_ledger_account and supplier_ledger_account.get("isInactive"):
            supplier_ledger_account = None

        if supplier_ledger_account is None:
            vendor_accounts = tripletex_client.get(
                "/ledger/account",
                params={
                    "ledgerType": "VENDOR",
                    "fields": "id,number,name,isInactive",
                    "count": 1000,
                },
            )
            vendor_accounts.raise_for_status()
            vendor_accounts = vendor_accounts.json()["values"]
            active_vendor_accounts = [
                x for x in vendor_accounts if not x.get("isInactive", False)
            ]
            active_vendor_accounts = sorted(
                active_vendor_accounts, key=lambda x: int(x["number"])
            )
            if not active_vendor_accounts:
                raise ValueError("No active vendor ledger account found")
            supplier_ledger_account = active_vendor_accounts[0]

        supplier_ledger_account

        voucher_payload = {
            "date": invoice_date,
            "description": f"Supplier invoice {self.invoice_number}",
            "postings": [
                {
                    "row": 1,
                    "date": invoice_date,
                    "description": f"Supplier invoice {self.invoice_number}",
                    "amountGross": -float(self.amount_incl_vat),
                    "amountGrossCurrency": -float(self.amount_incl_vat),
                    "account": {"id": supplier_ledger_account["id"]},
                    "supplier": {"id": supplier["id"]},
                    "invoiceNumber": self.invoice_number,
                    "termOfPayment": due_date,
                },
                {
                    "row": 2,
                    "date": invoice_date,
                    "description": self.description,
                    "amountGross": float(self.amount_incl_vat),
                    "amountGrossCurrency": float(self.amount_incl_vat),
                    "account": {"id": expense_account["id"]},
                    "vatType": {"id": vat_type["id"]},
                },
            ],
        }
        voucher_payload

        voucher_r = tripletex_client.post(
            "/ledger/voucher",
            params={"sendToLedger": "true"},
            json=voucher_payload,
        )
        voucher = voucher_r.json()
        voucher

        supplier_invoice_r = tripletex_client.get(
            "/supplierInvoice",
            params={
                "voucherId": voucher["id"],
                "invoiceDateFrom": invoice_date,
                "invoiceDateTo": (
                    date.fromisoformat(invoice_date) + timedelta(days=1)
                ).isoformat(),
                "fields": "id,invoiceNumber,invoiceDate,invoiceDueDate,supplier(id,name),voucher(id,number)",
                "count": 1000,
            },
        )
        supplier_invoice_r.raise_for_status()
        supplier_invoice_r = supplier_invoice_r.json()["values"]

        (supplier_invoice,) = [
            x for x in supplier_invoice_r if x["voucher"]["id"] == voucher["id"]
        ]
        supplier_invoice


if __name__ == "__main__":
    pass
    # from pathlib import Path
    #
    # for attachment_path in attachments:
    #     attachment = Path(attachment_path).read_text()
    #     res = SupplierInvoiceFromPDF.parse(prompts[0], attachment)
    #     print(f"file={attachment_path}")
    #     print(f"{res=}")
    #     print()
    #     break
