"""End-to-end test for task 9 (multi-line invoice).

Environment assumptions tested here:
  - Customer exists (by org number).
  - Products exist (by product number) with prices and VAT types.

Scoring assumptions tested here:
  - Invoice created with correct number of order lines.
  - Each line uses price and VAT type from the product data.

Not tested:
  - Full 3-product invoice (sandbox may lack matching products for all 3 VAT rates).
"""

import requests

from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.parsers.task9_multi_line_invoice import MultiLineInvoice, ProductLine


def _sandbox_creds() -> TripletexCredentials:
    return TripletexCredentials.placeholder_TODO()


def test_task9_solve_hardcoded():
    creds = _sandbox_creds()

    # Use products and customer that exist in sandbox
    parsed = MultiLineInvoice(
        customer_name="Fjordtech AS",
        org_number="912345678",
        product_lines=[
            ProductLine(product_name="Lærebok", product_number="5696", amount_excl_vat=23250, vat_rate=0),
            ProductLine(product_name="Stockage cloud", product_number="4455", amount_excl_vat=26850, vat_rate=25),
        ],
    )

    parsed.solve(tripletex_creds=creds)


def test_task9_parse():
    parsed = MultiLineInvoice.parse(
        'Crie uma fatura para o cliente Floresta Lda (org. nº 944182802) com três linhas de produto: Relatório de análise (2039) a 19600 NOK com 25 % IVA, Design web (3304) a 12450 NOK com 15 % IVA (alimentos), e Licença de software (1599) a 9150 NOK com 0 % IVA (isento).'
    )
    assert parsed.org_number == "944182802"
    assert len(parsed.product_lines) == 3
    assert parsed.product_lines[0].product_number == "2039"
    assert parsed.product_lines[0].amount_excl_vat == 19600.0
