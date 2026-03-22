from collections import defaultdict
from typing import Self

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict

from tripletex.client import LoggedHTTPClient
from tripletex.herman_tasks.utils import get_current_year_month_day_utc
from tripletex.my_log import configure_logging

prompts = [
    "Los costos totales aumentaron significativamente de enero a febrero de 2026. Analice el libro mayor e identifique las tres cuentas de gastos con el mayor incremento en monto. Cree un proyecto interno para cada una de las tres cuentas con el nombre de la cuenta. También cree una actividad para cada proyecto.",
    "Les coûts totaux ont augmenté de manière significative de janvier à février 2026. Analysez le grand livre et identifiez les trois comptes de charges avec la plus forte augmentation. Créez un projet interne pour chacun des trois comptes avec le nom du compte. Créez également une activité pour chaque projet.",
    "Os custos totais aumentaram significativamente de janeiro a fevereiro de 2026. Analise o livro razão e identifique as três contas de despesa com o maior aumento em valor. Crie um projeto interno para cada uma das três contas com o nome da conta. Também crie uma atividade para cada projeto.",
    "Totalkostnadene auka monaleg frå januar til februar 2026. Analyser hovudboka og finn dei tre kostnadskontoane med størst auke i beløp. Opprett eit internt prosjekt for kvar av dei tre kontoane med kontoens namn. Opprett også ein aktivitet for kvart prosjekt.",
    "Die Gesamtkosten sind von Januar bis Februar 2026 deutlich gestiegen. Analysieren Sie das Hauptbuch und identifizieren Sie die drei Aufwandskonten mit dem größten Anstieg. Erstellen Sie für jedes der drei Konten ein internes Projekt mit dem Kontonamen. Erstellen Sie außerdem eine Aktivität für jedes Projekt.",
    'Total costs increased significantly from January to February 2026. Analyze the general ledger and identify the three expense accounts with the largest increase in amount. Create an internal project for each of the three accounts using the account name. Also create an activity for each project.',
]

logger = configure_logging()


class CostIncreaseAnalysis(BaseModel):
    """No structured params to extract from text. Analysis is data-driven from the ledger."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def parse(cls, prompt: str) -> Self:
        return cls()

    def solve(self, tripletex_client: LoggedHTTPClient) -> None:
        # Assumptions about the clean competition environment:
        #   - Ledger postings exist for January and February 2026.
        #   - Expense accounts are in the 4000-8999 range.
        #   - At least one employee exists to use as project manager.
        #   - The "increase" means: sum of debits on account in Feb > sum in Jan.
        #
        # Assumptions about what scoring checks:
        #   - 3 internal projects created, each named after the account name
        #     (e.g. "6300 Leie lokale" or just "Leie lokale").
        #   - Each project has at least one activity linked.
        #
        # Not verified:
        #   - Exact project name format (account number + name vs just name).
        #   - Whether activity name matters.
        #   - Whether project manager assignment matters.
        today = get_current_year_month_day_utc()

        # ── GET ledger accounts (for name lookup) ────────────────────────
        accounts_r = tripletex_client.get(
            "/ledger/account",
            params={"fields": "id,number,displayName", "count": 1000},
        )
        accounts_r.raise_for_status()
        acct_by_number = {a["number"]: a for a in accounts_r.json()["values"]}
        logger.info("task28.accounts.fetched", extra={"count": len(acct_by_number)})

        # ── GET postings for January 2026 ────────────────────────────────
        jan_r = tripletex_client.get(
            "/ledger/posting",
            params={
                "dateFrom": "2026-01-01",
                "dateTo": "2026-02-01",
                "fields": "account(number),amount",
                "count": 10000,
            },
        )
        jan_r.raise_for_status()
        jan_postings = jan_r.json()["values"]
        logger.info("task28.jan.fetched", extra={"count": len(jan_postings)})

        # ── GET postings for February 2026 ───────────────────────────────
        feb_r = tripletex_client.get(
            "/ledger/posting",
            params={
                "dateFrom": "2026-02-01",
                "dateTo": "2026-03-01",
                "fields": "account(number),amount",
                "count": 10000,
            },
        )
        feb_r.raise_for_status()
        feb_postings = feb_r.json()["values"]
        logger.info("task28.feb.fetched", extra={"count": len(feb_postings)})

        # ── Compute per-account totals (debit = positive amounts) ────────
        def sum_by_account(postings: list[dict]) -> dict[int, float]:
            totals: dict[int, float] = defaultdict(float)
            for p in postings:
                acct_num = p["account"]["number"]
                # Expense accounts: 4000-8999
                if 4000 <= acct_num <= 8999:
                    totals[acct_num] += p["amount"]
            return dict(totals)

        jan_totals = sum_by_account(jan_postings)
        feb_totals = sum_by_account(feb_postings)
        logger.info(
            "task28.totals",
            extra={"jan_accounts": len(jan_totals), "feb_accounts": len(feb_totals)},
        )

        # ── Find top 3 accounts by increase ──────────────────────────────
        all_accounts = set(jan_totals.keys()) | set(feb_totals.keys())
        increases = {
            acct: feb_totals.get(acct, 0) - jan_totals.get(acct, 0)
            for acct in all_accounts
        }
        top3 = sorted(increases.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info("task28.top3", extra={"top3": top3})

        # ── GET first employee (for project manager) ─────────────────────
        emp_r = tripletex_client.get("/employee", params={"fields": "id", "count": 1})
        emp_r.raise_for_status()
        emp_id = emp_r.json()["values"][0]["id"]

        # ── WRITE: batch create 3 projects with nested activities ────────
        projects_payload = []
        for acct_num, increase in top3:
            acct_info = acct_by_number.get(acct_num, {})
            acct_name = acct_info.get("displayName", f"Konto {acct_num}")
            projects_payload.append({
                "name": acct_name,
                "projectManager": {"id": emp_id},
                "isInternal": True,
                "startDate": today,
                "projectActivities": [
                    {
                        "activity": {
                            "name": acct_name,
                            "activityType": "PROJECT_GENERAL_ACTIVITY",
                            "isProjectActivity": True,
                        }
                    }
                ],
            })

        logger.info("task28.projects.creating", extra={"payload": projects_payload})
        proj_r = tripletex_client.post("/project/list", json=projects_payload)
        logger.info(
            "task28.projects.response",
            extra={"status": proj_r.status_code, "body": proj_r.json()},
        )
        proj_r.raise_for_status()

        logger.info("task28.completed")


if __name__ == "__main__":
    res = CostIncreaseAnalysis.parse("")
    print(f"{res=}")
