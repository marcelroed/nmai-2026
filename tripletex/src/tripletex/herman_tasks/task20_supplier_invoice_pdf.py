from datetime import date, timedelta

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.my_log import configure_logging

logger = configure_logging()

# +

self_tripletex_creds = TripletexCredentials.placeholder_TODO()

self_supplier_name: str = "Bergvik AS"
self_org_number: str = "999001234"
self_supplier_street_address: str = "Sjogata 2"
self_supplier_postal_code: str = "1482"
self_supplier_city: str = "Nittedal"
self_supplier_bank_account: str = "58637944698"
self_invoice_number: str = "INV-2026-8506"
self_invoice_year: int = 2026
self_invoice_month: int = 2
self_invoice_day: int = 1
self_due_year: int = 2026
self_due_month: int = 3
self_due_day: int = 3
self_description: str = "Kontorrekvisita"
self_amount_excl_vat: float = 41050
self_vat_rate: float = 0.25
self_amount_incl_vat: float = 51312
self_expense_account = "6500"

# +

invoice_date = date(self_invoice_year, self_invoice_month, self_invoice_day).isoformat()
due_date = date(self_due_year, self_due_month, self_due_day).isoformat()

# +

suppliers = requests.get(
    f"{self_tripletex_creds.base_url}/supplier",
    auth=self_tripletex_creds.auth,
    params={
        "fields": "id,name,organizationNumber,ledgerAccount(id,number,name,isInactive)",
        "count": 1000,
    },
)
suppliers.raise_for_status()
suppliers = suppliers.json()["values"]

existing_supplier = [
    x for x in suppliers if x.get("organizationNumber") == self_org_number
]
if len(existing_supplier) > 1:
    raise ValueError(
        f"Expected at most one supplier for org number {self_org_number}, got {len(existing_supplier)}"
    )

# +

if existing_supplier:
    logger.info(
        "using existing_supplier", extra={"existing_supplier": existing_supplier}
    )
    (supplier,) = existing_supplier
    logger.info("picked the only possible supplier", extra={"supplier": supplier})
else:
    supplier_payload = {
        "name": self_supplier_name,
        "organizationNumber": self_org_number,
        "postalAddress": {
            "addressLine1": self_supplier_street_address,
            "postalCode": self_supplier_postal_code,
            "city": self_supplier_city,
        },
    }
    supplier_post_r = requests.post(
        f"{self_tripletex_creds.base_url}/supplier",
        auth=self_tripletex_creds.auth,
        json=supplier_payload,
    )
    supplier_post_r.raise_for_status()
    supplier = supplier_post_r.json()["value"]

supplier

# +

ledger_accounts = requests.get(
    f"{self_tripletex_creds.base_url}/ledger/account",
    auth=self_tripletex_creds.auth,
    params={
        "number": str(self_expense_account),
        "isApplicableForSupplierInvoice": "true",
        "fields": "id,number,name,ledgerType,isApplicableForSupplierInvoice,vatLocked,vatType(id,number),legalVatTypes(id,number)",
        "count": 1000,
    },
)
ledger_accounts.raise_for_status()
ledger_accounts = ledger_accounts.json()["values"]

(expense_account,) = [
    x for x in ledger_accounts if str(x["number"]) == str(self_expense_account)
]

if not expense_account["isApplicableForSupplierInvoice"]:
    raise ValueError(
        f"Expense account {self_expense_account} is not applicable for supplier invoices"
    )

expense_account

# +

# TODO: check that this should actually be
incoming_vat_types = requests.get(
    f"{self_tripletex_creds.base_url}/ledger/vatType",
    auth=self_tripletex_creds.auth,
    params={
        "typeOfVat": "INCOMING",
        "number": 1,
        "fields": "id,number,name",
        "count": 1000,
    },
)
incoming_vat_types.raise_for_status()
incoming_vat_types = incoming_vat_types.json()["values"]

(incoming_vat_type,) = [x for x in incoming_vat_types if x["number"] == 1]

vat_type = (
    expense_account["vatType"] if expense_account["vatLocked"] else incoming_vat_type
)

legal_vat_type_ids = {x["id"] for x in expense_account.get("legalVatTypes", [])}
if legal_vat_type_ids and vat_type["id"] not in legal_vat_type_ids:
    raise ValueError(
        f"VAT type {vat_type['id']} is not legal for expense account {expense_account['number']}"
    )

vat_type

# +

supplier_ledger_account = supplier.get("ledgerAccount")
if supplier_ledger_account and supplier_ledger_account.get("isInactive"):
    supplier_ledger_account = None

if supplier_ledger_account is None:
    vendor_accounts = requests.get(
        f"{self_tripletex_creds.base_url}/ledger/account",
        auth=self_tripletex_creds.auth,
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

# +

voucher_payload = {
    "date": invoice_date,
    "description": f"Supplier invoice {self_invoice_number}",
    "postings": [
        {
            "row": 1,
            "date": invoice_date,
            "description": f"Supplier invoice {self_invoice_number}",
            "amountGross": -float(self_amount_incl_vat),
            "amountGrossCurrency": -float(self_amount_incl_vat),
            "account": {"id": supplier_ledger_account["id"]},
            "supplier": {"id": supplier["id"]},
            "invoiceNumber": self_invoice_number,
            "termOfPayment": due_date,
        },
        {
            "row": 2,
            "date": invoice_date,
            "description": self_description,
            "amountGross": float(self_amount_incl_vat),
            "amountGrossCurrency": float(self_amount_incl_vat),
            "account": {"id": expense_account["id"]},
            "vatType": {"id": vat_type["id"]},
        },
    ]
}
voucher_payload

# +

voucher_r = requests.post(
    f"{self_tripletex_creds.base_url}/ledger/voucher",
    auth=self_tripletex_creds.auth,
    params={"sendToLedger": "true"},
    json=voucher_payload,
)
voucher = voucher_r.json()
voucher

# +

supplier_invoice_r = requests.get(
    f"{self_tripletex_creds.base_url}/supplierInvoice",
    auth=self_tripletex_creds.auth,
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
