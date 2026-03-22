from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import (
    get_customer_by_org_number,
    get_invoice_by_amount_excluding_vat,
)
from tripletex.my_log import configure_logging

prompts = [
    'Le paiement de Colline SARL (nº org. 916057903) pour la facture "Maintenance" (8300 NOK HT) a été retourné par la banque. Annulez le paiement afin que la facture affiche à nouveau le montant impayé.',
    'The payment from Ridgepoint Ltd (org no. 990845042) for the invoice "Cloud Storage" (43550 NOK excl. VAT) was returned by the bank. Reverse the payment so the invoice shows the outstanding amount again.',
    'Die Zahlung von Brückentor GmbH (Org.-Nr. 944848479) für die Rechnung "Wartung" (42200 NOK ohne MwSt.) wurde von der Bank zurückgebucht. Stornieren Sie die Zahlung, damit die Rechnung wieder den offenen Betrag anzeigt.',
    'Die Zahlung von Nordlicht GmbH (Org.-Nr. 985405077) für die Rechnung "Wartung" (42200 NOK ohne MwSt.) wurde von der Bank zurückgebucht. Stornieren Sie die Zahlung, damit die Rechnung wieder den offenen Betrag anzeigt.',
    'Le paiement de Étoile SARL (nº org. 943745862) pour la facture "Conseil en données" (33900 NOK HT) a été retourné par la banque. Annulez le paiement afin que la facture affiche à nouveau le montant impayé.',
    'Betalingen fra Polaris AS (org.nr 896496468) for fakturaen "Skylagring" (17200 kr ekskl. MVA) ble returnert av banken. Reverser betalingen slik at fakturaen igjen viser utestående beløp.',
    'El pago de Solmar SL (org. nº 836598741) por la factura "Licencia de software" (8000 NOK sin IVA) fue devuelto por el banco. Revierta el pago para que la factura vuelva a mostrar el importe pendiente.',
    'Le paiement de Lumière SARL (nº org. 937075405) pour la facture "Conseil en données" (39100 NOK HT) a été retourné par la banque. Annulez le paiement afin que la facture affiche à nouveau le montant impayé.',
]

logger = configure_logging()


class ReversePayment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    invoice_description: str
    amount_excl_vat: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=ReversePayment,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        customer = get_customer_by_org_number(
            org_number=self.org_number, tripletex_client=tripletex_client
        )

        invoice = get_invoice_by_amount_excluding_vat(
            amount_excluding_vat=self.amount_excl_vat, tripletex_client=tripletex_client
        )
        invoice

        logger.info(
            "task18.customer_match",
            extra={"matches_customer": invoice["customer"]["id"] == customer["id"]},
        )
        invoice["postings"]

        postings = tripletex_client.get(
            "/ledger/posting",
            params={"dateFrom": "1970-01-01", "dateTo": "2027-12-31"},
        ).json()
        postings

        invoice_posting_ids = [posting["id"] for posting in invoice["postings"]]
        invoice_posting_ids
        (correct_posting,) = [
            x
            for x in postings["values"]
            if x["id"] in invoice_posting_ids
            and x["type"] == "OUTGOING_INVOICE_CUSTOMER_POSTING"
        ]

        reverse_voucher_r = tripletex_client.put(
            f"/ledger/voucher/{invoice['voucher']['id']}/:reverse",
            params={"date": correct_posting["date"]},
        )
        reverse_voucher_r.json()


if __name__ == "__main__":
    pass
    # for prompt in prompts:
    #     res = ReversePayment.parse(prompt)
    #     print(f"{prompt=}")
    #     print(f"{res=}")
    #     print()
