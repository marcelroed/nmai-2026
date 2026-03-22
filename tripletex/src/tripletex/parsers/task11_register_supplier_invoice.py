from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient

prompts = [
    "We have received invoice INV-2026-9075 from the supplier Brightstone Ltd (org no. 890932991) for 59800 NOK including VAT. The amount relates to office services (account 6300). Register the supplier invoice with the correct input VAT (25%).",
    "We have received invoice INV-2026-3749 from the supplier Ridgepoint Ltd (org no. 902484981) for 65850 NOK including VAT. The amount relates to office services (account 6590). Register the supplier invoice with the correct input VAT (25%).",
    "Recebemos a fatura INV-2026-6556 do fornecedor Solmar Lda (org. nº 974178680) no valor de 50750 NOK com IVA incluído. O montante refere-se a serviços de escritório (conta 6500). Registe a fatura do fornecedor com o IVA dedutível correto (25 %).",
    "Nous avons reçu la facture INV-2026-4647 du fournisseur Cascade SARL (nº org. 951586935) de 30000 NOK TTC. Le montant concerne des services de bureau (compte 6860). Enregistrez la facture fournisseur avec la TVA déductible correcte (25 %).",
    "We have received invoice INV-2026-8735 from the supplier Brightstone Ltd (org no. 913701585) for 8500 NOK including VAT. The amount relates to office services (account 7100). Register the supplier invoice with the correct input VAT (25%).",
    "We have received invoice INV-2026-3205 from the supplier Ironbridge Ltd (org no. 828254375) for 24500 NOK including VAT. The amount relates to office services (account 6590). Register the supplier invoice with the correct input VAT (25%).",
    "Recebemos a fatura INV-2026-6293 do fornecedor Montanha Lda (org. nº 980979431) no valor de 12050 NOK com IVA incluído. O montante refere-se a serviços de escritório (conta 7000). Registe a fatura do fornecedor com o IVA dedutível correto (25 %).",
    "Recebemos a fatura INV-2026-4855 do fornecedor Solmar Lda (org. nº 972752843) no valor de 62600 NOK com IVA incluído. O montante refere-se a serviços de escritório (conta 6860). Registe a fatura do fornecedor com o IVA dedutível correto (25 %).",
    'Vi har mottatt faktura INV-2026-9382 fra leverandøren Stormberg AS (org.nr 877462137) på 61600 kr inklusiv MVA. Beløpet gjelder kontortjenester (konto 6340). Registrer leverandørfakturaen med korrekt inngående MVA (25 %).',
]


class RegisterSupplierInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: Annotated[str, Field(pattern=r"^INV-\d{4}-\d+$")]
    supplier_name: str
    org_number: Annotated[str, Field(pattern=r"^\d{9}$")]
    amount_incl_vat: float
    expense_account: Annotated[str, Field(pattern=r"^\d{4}$")]
    vat_rate: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=RegisterSupplierInvoice,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient):
        pass


if __name__ == "__main__":
    for prompt in prompts:
        res = RegisterSupplierInvoice.parse(prompt)
        print(f"{prompt=}")
        print(f"{res=}")
        print()
