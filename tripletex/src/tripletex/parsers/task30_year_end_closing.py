from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.my_log import configure_logging

prompts = [
    "Realize o encerramento anual simplificado de 2025: 1) Calcule e registe a depreciação anual de três ativos: Kontormaskiner (301750 NOK, 10 anos lineares, conta 1200), IT-utstyr (304100 NOK, 3 anos, conta 1210), Programvare (322800 NOK, 8 anos, conta 1250). Use conta 6010 para despesa de depreciação e 1209 para depreciação acumulada. 2) Reverta despesas antecipadas (total 27150 NOK na conta 1700). 3) Calcule e registe a provisão fiscal (22 % do resultado tributável) na conta 8700/2920. Registe cada depreciação como um lançamento separado.",
    "Führen Sie den vereinfachten Jahresabschluss für 2025 durch: 1) Berechnen und buchen Sie die jährliche Abschreibung für drei Anlagen: Programvare (419000 NOK, 6 Jahre linear, Konto 1250), Kontormaskiner (374150 NOK, 8 Jahre, Konto 1200), Kjøretøy (64300 NOK, 9 Jahre, Konto 1230). Verwenden Sie Konto 6010 für Abschreibungsaufwand und 1209 für kumulierte Abschreibungen. 2) Lösen Sie vorausbezahlte Aufwendungen auf (insgesamt 51000 NOK auf Konto 1700). 3) Berechnen und buchen Sie die Steuerrückstellung (22 % des steuerpflichtigen Gewinns) auf Konto 8700/2920. Buchen Sie jede Abschreibung als separaten Beleg.",
    "Perform simplified year-end closing for 2025: 1) Calculate and post annual depreciation for three assets: Kontormaskiner (138450 NOK, 6 years straight-line, account 1200), Programvare (280000 NOK, 9 years, account 1250), Inventar (484650 NOK, 8 years, account 1240). Use account 6010 for depreciation expense and 1209 for accumulated depreciation. 2) Reverse prepaid expenses (total 52250 NOK on account 1700). 3) Calculate and post tax provision (22% of taxable profit) on account 8700/2920. Post each depreciation as a separate voucher.",
    "Effectuez la clôture annuelle simplifiée pour 2025 : 1) Calculez et comptabilisez l'amortissement annuel de trois immobilisations : Programvare (111950 NOK, 9 ans linéaire, compte 1250), Kontormaskiner (351450 NOK, 9 ans, compte 1200), Inventar (418800 NOK, 10 ans, compte 1240). Utilisez le compte 6010 pour les charges d'amortissement et 1209 pour les amortissements cumulés. 2) Extournez les charges constatées d'avance (total 79750 NOK au compte 1700). 3) Calculez et comptabilisez la provision d'impôt (22 % du bénéfice imposable) sur compte 8700/2920. Comptabilisez chaque amortissement comme une pièce comptable séparée.",
    "Utfør forenklet årsoppgjør for 2025: 1) Beregn og bokfør årlige avskrivninger for tre eiendeler: Inventar (468300 kr, 10 år lineært, konto 1240), Kontormaskiner (149750 kr, 10 år, konto 1200), Kjøretøy (412950 kr, 9 år, konto 1230). Bruk konto 6010 for avskrivningskostnad og 1209 for akkumulerte avskrivninger. 2) Reverser forskuddsbetalte kostnader (totalt 47850 kr på konto 1700). 3) Beregn og bokfør skattekostnad (22 % av skattbart resultat) på konto 8700/2920. Bokfør hver avskrivning som et eget bilag.",
    "Führen Sie den vereinfachten Jahresabschluss für 2025 durch: 1) Berechnen und buchen Sie die jährliche Abschreibung für drei Anlagen: IT-utstyr (229850 NOK, 10 Jahre linear, Konto 1210), Kontormaskiner (146050 NOK, 8 Jahre, Konto 1200), Kjøretøy (311650 NOK, 6 Jahre, Konto 1230). Verwenden Sie Konto 6010 für Abschreibungsaufwand und 1209 für kumulierte Abschreibungen. 2) Lösen Sie vorausbezahlte Aufwendungen auf (insgesamt 27650 NOK auf Konto 1700). 3) Berechnen und buchen Sie die Steuerrückstellung (22 % des steuerpflichtigen Gewinns) auf Konto 8700/2920. Buchen Sie jede Abschreibung als separaten Beleg.",
    "Realice el cierre anual simplificado de 2025: 1) Calcule y contabilice la depreciación anual de tres activos: Programvare (359300 NOK, 4 años lineales, cuenta 1250), IT-utstyr (444050 NOK, 6 años, cuenta 1210), Kjøretøy (344650 NOK, 3 años, cuenta 1230). Use cuenta 6010 para gasto de depreciación y 1209 para depreciación acumulada. 2) Revierta gastos prepagados (total 36150 NOK en cuenta 1700). 3) Calcule y contabilice la provisión de impuestos (22 % del resultado imponible) en cuenta 8700/2920. Registre cada depreciación como un comprobante separado.",
    "Perform simplified year-end closing for 2025: 1) Calculate and post annual depreciation for three assets: IT-utstyr (204150 NOK, 4 years straight-line, account 1210), Inventar (237550 NOK, 8 years, account 1240), Programvare (307500 NOK, 4 years, account 1250). Use account 6010 for depreciation expense and 1209 for accumulated depreciation. 2) Reverse prepaid expenses (total 44300 NOK on account 1700). 3) Calculate and post tax provision (22% of taxable profit) on account 8700/2920. Post each depreciation as a separate voucher.",
    'Effectuez la clôture annuelle simplifiée pour 2025 : 1) Calculez et comptabilisez l\'amortissement annuel de trois immobilisations : Kontormaskiner (433350 NOK, 3 ans linéaire, compte 1200), IT-utstyr (491450 NOK, 6 ans, compte 1210), Kjøretøy (207050 NOK, 9 ans, compte 1230). Utilisez le compte 6010 pour les charges d\'amortissement et 1209 pour les amortissements cumulés. 2) Extournez les charges constatées d\'avance (total 63200 NOK au compte 1700). 3) Calculez et comptabilisez la provision d\'impôt (22 % du bénéfice imposable) sur compte 8700/2920. Comptabilisez chaque amortissement comme une pièce comptable séparée.',
]

logger = configure_logging()


class DepreciationAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_name: str
    cost: float
    useful_life_years: int
    asset_account: Annotated[int, Field(description="Asset account number (e.g. 1200, 1210, 1230, 1240, 1250)")]


class YearEndClosing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: list[DepreciationAsset]
    prepaid_expense_total: float

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=YearEndClosing,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Standard chart of accounts: 6010 (depreciation expense), 1209
        #     (accumulated depreciation), 1700 (prepaid), 6300 (rent expense),
        #     8700 (tax expense), 2920 (tax payable).
        #   - Accounts 1209 and 8700 exist in competition (missing from our sandbox).
        #   - Existing P&L postings for 2025 determine the taxable profit base.
        #   - Always exactly 3 assets.
        #
        # Assumptions about what scoring checks:
        #   - 3 separate depreciation vouchers, each: debit 6010, credit 1209,
        #     amount = cost / useful_life_years, dated 2025-12-31.
        #   - Prepaid reversal: debit 6300, credit 1700 for prepaid_expense_total.
        #   - Tax provision: debit 8700, credit 2920 for 22% of taxable profit
        #     (profit computed AFTER depreciations and prepaid reversal).
        #
        # Not verified:
        #   - Whether prepaid expense always goes to 6300 (depends on what was prepaid).
        #   - Whether taxable profit = sum of P&L or uses a different calculation.
        #   - Rounding of depreciation and tax amounts.
        posting_date = "2025-12-31"

        # ── GET ledger accounts ──────────────────────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}
        logger.info("task30.accounts.fetched", extra={"count": len(accts)})

        # ── GET current P&L for 2025 (before our postings) ──────────────
        bal_r = tripletex_client.get(
            "/balanceSheet",
            params={
                "dateFrom": "2025-01-01",
                "dateTo": "2025-12-31",
                "fields": "account(number),balanceChange",
                "count": 1000,
            },
        )
        bal_r.raise_for_status()
        balance_entries = bal_r.json()["values"]

        # P&L: income (3xxx) is credit-negative, expenses (4xxx-8xxx) are debit-positive
        # Net result = sum of all 3xxx-8xxx balanceChange (negative = profit)
        current_pl = sum(
            e["balanceChange"]
            for e in balance_entries
            if 3000 <= e["account"]["number"] <= 8999
        )
        logger.info("task30.currentPL", extra={"current_pl": current_pl})

        # ── Compute depreciation amounts ─────────────────────────────────
        dep_amounts = []
        total_depreciation = 0.0
        for asset in self.assets:
            dep = round(asset.cost / asset.useful_life_years, 2)
            dep_amounts.append(dep)
            total_depreciation += dep
        logger.info(
            "task30.depreciation",
            extra={
                "amounts": dep_amounts,
                "total": total_depreciation,
            },
        )

        # ── Compute taxable profit (after all adjustments) ───────────────
        # current_pl is negative for profit (credit-side income > debit-side expenses)
        # After posting: add depreciation expenses and prepaid reversal (both increase expenses)
        adjusted_pl = current_pl + total_depreciation + self.prepaid_expense_total
        taxable_profit = -adjusted_pl  # flip sign: positive = profit
        tax_amount = round(max(0, taxable_profit) * 0.22, 2)
        logger.info(
            "task30.tax",
            extra={
                "adjusted_pl": adjusted_pl,
                "taxable_profit": taxable_profit,
                "tax_amount": tax_amount,
            },
        )

        # ── WRITE 1-3: separate depreciation vouchers ────────────────────
        for i, (asset, dep_amount) in enumerate(zip(self.assets, dep_amounts)):
            dep_voucher = {
                "date": posting_date,
                "description": f"Avskrivning {asset.asset_name}",
                "postings": [
                    {
                        "account": {"id": accts[6010]["id"]},
                        "amountCurrency": dep_amount,
                        "amountGross": dep_amount,
                        "amountGrossCurrency": dep_amount,
                        "row": 1,
                    },
                    {
                        "account": {"id": accts[1209]["id"]},
                        "amountCurrency": -dep_amount,
                        "amountGross": -dep_amount,
                        "amountGrossCurrency": -dep_amount,
                        "row": 2,
                    },
                ],
            }
            logger.info(
                f"task30.depreciation.{i}.creating",
                extra={"asset": asset.asset_name, "amount": dep_amount, "payload": dep_voucher},
            )
            r = tripletex_client.post("/ledger/voucher", json=dep_voucher)
            logger.info(
                f"task30.depreciation.{i}.response",
                extra={"status": r.status_code, "body": r.json()},
            )
            r.raise_for_status()

        # ── WRITE 4: prepaid reversal + tax provision (combined) ─────────
        closing_postings = []
        row = 1

        # Prepaid reversal: debit 6300 (expense), credit 1700 (prepaid)
        closing_postings.append({
            "account": {"id": accts[6300]["id"]},
            "amountCurrency": self.prepaid_expense_total,
            "amountGross": self.prepaid_expense_total,
            "amountGrossCurrency": self.prepaid_expense_total,
            "row": row,
        })
        row += 1
        closing_postings.append({
            "account": {"id": accts[1700]["id"]},
            "amountCurrency": -self.prepaid_expense_total,
            "amountGross": -self.prepaid_expense_total,
            "amountGrossCurrency": -self.prepaid_expense_total,
            "row": row,
        })
        row += 1

        # Tax provision: debit 8700 (tax expense), credit 2920 (tax payable)
        if tax_amount > 0:
            closing_postings.append({
                "account": {"id": accts[8700]["id"]},
                "amountCurrency": tax_amount,
                "amountGross": tax_amount,
                "amountGrossCurrency": tax_amount,
                "row": row,
            })
            row += 1
            closing_postings.append({
                "account": {"id": accts[2920]["id"]},
                "amountCurrency": -tax_amount,
                "amountGross": -tax_amount,
                "amountGrossCurrency": -tax_amount,
                "row": row,
            })

        closing_voucher = {
            "date": posting_date,
            "description": "Årsoppgjør 2025 - forskuddsbetalt og skatt",
            "postings": closing_postings,
        }
        logger.info("task30.closing.creating", extra={"payload": closing_voucher})
        r = tripletex_client.post("/ledger/voucher", json=closing_voucher)
        logger.info(
            "task30.closing.response",
            extra={"status": r.status_code, "body": r.json()},
        )
        r.raise_for_status()

        logger.info("task30.completed")


if __name__ == "__main__":
    for prompt in prompts[:1]:
        res = YearEndClosing.parse(prompt)
        print(f"{prompt[:80]}...")
        print(f"  {res=}")
        print()
