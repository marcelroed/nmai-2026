from typing import Annotated, Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from tripletex.client import LoggedHTTPClient
from tripletex.my_log import configure_logging

prompts = [
    "Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (4200 NOK pro Monat von Konto 1700 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 64000 NOK und Nutzungsdauer 3 Jahre (lineare Abschreibung auf Konto 6030). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).",
    "Realize o encerramento mensal de março de 2026. Registe a reversão de acréscimos (9700 NOK por mês da conta 1710 para despesa). Registe a depreciação mensal de um ativo fixo com custo de aquisição 117300 NOK e vida útil 3 anos (depreciação linear para conta 6020). Verifique se o balancete está a zero. Registe também uma provisão salarial (débito conta de despesas salariais 5000, crédito conta de salários acumulados 2900).",
    "Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (2200 NOK pro Monat von Konto 1720 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 291700 NOK und Nutzungsdauer 5 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).",
    "Realize o encerramento mensal de março de 2026. Registe a reversão de acréscimos (9000 NOK por mês da conta 1700 para despesa). Registe a depreciação mensal de um ativo fixo com custo de aquisição 289500 NOK e vida útil 9 anos (depreciação linear para conta 6030). Verifique se o balancete está a zero. Registe também uma provisão salarial (débito conta de despesas salariais 5000, crédito conta de salários acumulados 2900).",
    "Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (3500 NOK pro Monat von Konto 1710 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 79150 NOK und Nutzungsdauer 6 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).",
    "Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (6150 NOK pro Monat von Konto 1710 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 181400 NOK und Nutzungsdauer 3 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).",
    "Führen Sie den Monatsabschluss für März 2026 durch. Buchen Sie die Rechnungsabgrenzung (3400 NOK pro Monat von Konto 1700 auf Aufwand). Erfassen Sie die monatliche Abschreibung für eine Anlage mit Anschaffungskosten 289700 NOK und Nutzungsdauer 7 Jahre (lineare Abschreibung auf Konto 6020). Überprüfen Sie, ob die Saldenbilanz null ergibt. Buchen Sie außerdem eine Gehaltsrückstellung (Soll Gehaltsaufwand Konto 5000, Haben aufgelaufene Gehälter Konto 2900).",
    'Effectuez la clôture mensuelle de mars 2026. Comptabilisez la régularisation (13600 NOK par mois du compte 1700 vers charges). Enregistrez l\'amortissement mensuel d\'une immobilisation avec un coût d\'acquisition de 262850 NOK et une durée de vie utile de 10 ans (amortissement linéaire sur compte 6030). Vérifiez que la balance est à zéro. Comptabilisez également une provision pour salaires (débit compte de charges salariales 5000, crédit compte de salaires à payer 2900).',
    'Realice el cierre mensual de marzo de 2026. Registre la periodificación (11900 NOK por mes de la cuenta 1700 a gasto). Contabilice la depreciación mensual de un activo fijo con costo de adquisición 107950 NOK y vida útil 6 años (depreciación lineal a cuenta 6010). Verifique que el balance de saldos sea cero. También registre una provisión salarial (débito cuenta de gastos salariales 5000, crédito cuenta de salarios acumulados 2900).',
    'Realice el cierre mensual de marzo de 2026. Registre la periodificación (3500 NOK por mes de la cuenta 1700 a gasto). Contabilice la depreciación mensual de un activo fijo con costo de adquisición 232650 NOK y vida útil 6 años (depreciación lineal a cuenta 6030). Verifique que el balance de saldos sea cero. También registre una provisión salarial (débito cuenta de gastos salariales 5000, crédito cuenta de salarios acumulados 2900).',
]

logger = configure_logging()

# Standard mapping: prepaid account → expense account
ACCRUAL_EXPENSE_MAP = {
    1700: 6300,  # Forskuddsbetalt leiekostnad → Leie lokale
    1710: 8150,  # Forskuddsbetalt rentekostnad → Annen rentekostnad
    1720: 6490,  # Andre depositum → Annen leiekostnad
}

# Standard mapping: depreciation expense → accumulated depreciation (contra-asset)
# The credit goes to the asset account that is being depreciated.
# In a clean sandbox, we look up which asset account has a balance.
DEPRECIATION_CONTRA_MAP = {
    6000: 1100,  # Avskrivning bygninger → Forretningsbygg
    6010: 1230,  # Avskrivning transportmidler → Vare- og lastebiler
    6015: 1200,  # Avskrivning maskiner → Maskiner og anlegg
    6017: 1250,  # Avskrivning inventar → Inventar
    6020: 1080,  # Avskrivning immaterielle → Goodwill (or other intangible)
}


class MonthlyClosing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accrual_amount_per_month: float
    accrual_source_account: Annotated[
        int,
        Field(description="Prepaid account to reverse from (e.g. 1700, 1710, 1720)"),
    ]
    asset_cost: float
    useful_life_years: int
    depreciation_account: Annotated[
        int, Field(description="Depreciation expense account (e.g. 6020, 6030)")
    ]

    @classmethod
    def parse(cls, prompt: str) -> Self:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=MonthlyClosing,
        )
        return response.parsed_output

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Standard chart of accounts present.
        #   - The depreciation expense account exists (6020 always does; 6030 may
        #     need to be created or may exist in competition sandboxes).
        #   - Exactly one employee (or a known set) exists — total monthly salary
        #     is derived from employee/employment/details.
        #   - Accrual source account (1700/1710/1720) maps to a standard expense
        #     account via ACCRUAL_EXPENSE_MAP.
        #   - A fixed asset exists whose contra-account can be inferred from
        #     DEPRECIATION_CONTRA_MAP.
        #
        # Assumptions about what scoring checks:
        #   - Accrual voucher: debit expense, credit prepaid for accrual_amount_per_month.
        #   - Depreciation voucher: debit depreciation account, credit contra-asset
        #     for asset_cost / useful_life_years / 12 (rounded to 2 decimals).
        #   - Salary provision voucher: debit 5000, credit 2900 for total monthly salary.
        #   - All postings dated in March 2026.
        #
        # Not verified:
        #   - Whether the contra-asset account mapping is correct.
        #   - Whether salary provision amount should include employer contributions.
        #   - Whether rounding of depreciation matters (we round to 2 decimals).
        posting_date = "2026-03-31"

        # ── GET ledger accounts ──────────────────────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number", "count": 1000},
        )
        accounts_r.raise_for_status()
        accts = {a["number"]: a for a in accounts_r.json()["values"]}
        logger.info("task26.accounts.fetched", extra={"count": len(accts)})

        # ── GET total monthly salary from employment details ─────────────
        details_r = tripletex_client.get(
            "/employee/employment/details",
            params={"fields": "monthlySalary", "count": 1000},
        )
        details_r.raise_for_status()
        total_monthly_salary = sum(
            d["monthlySalary"]
            for d in details_r.json()["values"]
            if d["monthlySalary"] > 0
        )
        logger.info(
            "task26.salary.computed",
            extra={"total_monthly_salary": total_monthly_salary},
        )

        # ── Compute amounts ──────────────────────────────────────────────
        depreciation_monthly = round(self.asset_cost / self.useful_life_years / 12, 2)
        accrual_expense_account = ACCRUAL_EXPENSE_MAP.get(
            self.accrual_source_account, 6300
        )
        depreciation_contra = DEPRECIATION_CONTRA_MAP.get(
            self.depreciation_account, 1200
        )

        logger.info(
            "task26.amounts",
            extra={
                "accrual": self.accrual_amount_per_month,
                "accrual_expense_account": accrual_expense_account,
                "depreciation_monthly": depreciation_monthly,
                "depreciation_contra": depreciation_contra,
                "salary_provision": total_monthly_salary,
            },
        )

        # ── WRITE: single voucher with all 3 closing entries ─────────────
        postings = []
        row = 1

        # 1. Accrual reversal: debit expense, credit prepaid
        postings.append(
            {
                "account": {"id": accts[accrual_expense_account]["id"]},
                "amountCurrency": self.accrual_amount_per_month,
                "amountGross": self.accrual_amount_per_month,
                "amountGrossCurrency": self.accrual_amount_per_month,
                "row": row,
            }
        )
        row += 1
        postings.append(
            {
                "account": {"id": accts[self.accrual_source_account]["id"]},
                "amountCurrency": -self.accrual_amount_per_month,
                "amountGross": -self.accrual_amount_per_month,
                "amountGrossCurrency": -self.accrual_amount_per_month,
                "row": row,
            }
        )
        row += 1

        # 2. Depreciation: debit depreciation expense, credit contra-asset
        postings.append(
            {
                "account": {"id": accts[self.depreciation_account]["id"]},
                "amountCurrency": depreciation_monthly,
                "amountGross": depreciation_monthly,
                "amountGrossCurrency": depreciation_monthly,
                "row": row,
            }
        )
        row += 1
        postings.append(
            {
                "account": {"id": accts[depreciation_contra]["id"]},
                "amountCurrency": -depreciation_monthly,
                "amountGross": -depreciation_monthly,
                "amountGrossCurrency": -depreciation_monthly,
                "row": row,
            }
        )
        row += 1

        # 3. Salary provision: debit 5000, credit 2900
        postings.append(
            {
                "account": {"id": accts[5000]["id"]},
                "amountCurrency": total_monthly_salary,
                "amountGross": total_monthly_salary,
                "amountGrossCurrency": total_monthly_salary,
                "row": row,
            }
        )
        row += 1
        postings.append(
            {
                "account": {"id": accts[2900]["id"]},
                "amountCurrency": -total_monthly_salary,
                "amountGross": -total_monthly_salary,
                "amountGrossCurrency": -total_monthly_salary,
                "row": row,
            }
        )

        voucher_payload = {
            "date": posting_date,
            "description": "Månedsslutt mars 2026",
            "postings": postings,
        }
        logger.info("task26.voucher.creating", extra={"payload": voucher_payload})
        r = tripletex_client.post("/ledger/voucher", json=voucher_payload)
        logger.info(
            "task26.voucher.response",
            extra={"status": r.status_code, "body": r.json()},
        )
        r.raise_for_status()

        logger.info("task26.completed")


if __name__ == "__main__":
    for prompt in prompts:
        res = MonthlyClosing.parse(prompt)
        print(f"{prompt[:80]}...")
        print(f"  {res=}")
        print()
