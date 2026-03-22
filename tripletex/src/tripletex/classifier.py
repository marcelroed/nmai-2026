from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict

TASK_DESCRIPTIONS = """
Task 1: Create employee — new employee with name, date of birth, email, start date
Task 2: Create customer — customer with org number (9 digits), street address, postal code, city, email
Task 3: Create product — product with product number, price excl VAT, VAT rate
Task 4: Register supplier — supplier with org number and email (no PDF attachment)
Task 5: Create departments — list of department names to create
Task 6: Create and send invoice — single-line invoice to customer with amount excl VAT and product description
Task 7: Register payment — register full payment on an existing customer invoice (customer has outstanding/pending invoice)
Task 8: Create project — create project linked to customer with project manager name and email
Task 9: Multi-line invoice — invoice with 3 product lines at different VAT rates (25%, 15%, 0%)
Task 10: Order to invoice — create order with products, convert to invoice, register full payment
Task 11: Register supplier invoice — received invoice INV-YYYY-NNNN from supplier, amount including VAT, expense account
Task 12: Payroll with bonus — run payroll for employee with base salary and one-time bonus
Task 13: Travel expense — travel expense with per diem (daily rate), flight ticket, and taxi costs
Task 14: Credit note — customer complained, issue full credit note reversing entire invoice
Task 15: Fixed price project — set fixed price on project, invoice customer X% as milestone payment
Task 16: Time logging invoice — log hours on activity/project at hourly rate (NOK/h), generate project invoice
Task 17: Custom dimension — create accounting dimension with values, post voucher linked to dimension value
Task 18: Reverse payment — payment returned by bank, reverse it so invoice shows outstanding amount again
Task 19: Employee from contract PDF — create employee from attached EMPLOYMENT CONTRACT (arbeidskontrakt, contrat de travail, Arbeitsvertrag, contrato de trabalho). Keywords: personnummer, national identity number, occupation code, stillingskode
Task 20: Supplier invoice from PDF — register supplier invoice from attached PDF (create supplier if needed)
Task 21: Onboarding from offer letter PDF — complete onboarding from attached OFFER LETTER (tilbudsbrev, lettre d'offre, Angebotsschreiben, carta de oferta). Keywords: onboarding, integration, stillingsprosent, work hours
Task 22: Expense from receipt PDF — post specific expense item from attached receipt to a department
Task 23: Bank reconciliation CSV — reconcile attached bank statement CSV against open invoices
Task 24: Ledger error correction — find 4 errors in general ledger: wrong account, duplicate voucher, missing VAT line, incorrect amount
Task 25: Overdue reminder — find overdue invoice, post reminder fee (debit 1500/credit 3400), send reminder invoice, register partial payment of 5000
Task 26: Monthly closing — accrual reversal, monthly depreciation, salary provision (March 2026)
Task 27: Currency invoice — EUR invoice with exchange rate difference (agio/disagio)
Task 28: Cost increase analysis — analyze general ledger for 3 expense accounts with largest increase, create internal projects
Task 29: Complete project lifecycle — 4 numbered steps: budget, log time (PM + consultant), supplier cost, customer invoice
Task 30: Year-end closing — annual depreciation for 3 assets, reverse prepaid expenses, tax provision (22%)
""".strip()

SYSTEM_PROMPT = f"""You are a task classifier for a Norwegian accounting system (Tripletex).
Given a user prompt (which may be in Norwegian, English, German, French, Spanish, or Portuguese),
classify it to exactly one of the 30 task types below.

{TASK_DESCRIPTIONS}

Return only the task number."""

type Task = Literal[
    "Task 1",
    "Task 2",
    "Task 3",
    "Task 4",
    "Task 5",
    "Task 6",
    "Task 7",
    "Task 8",
    "Task 9",
    "Task 10",
    "Task 11",
    "Task 12",
    "Task 13",
    "Task 14",
    "Task 15",
    "Task 16",
    "Task 17",
    "Task 18",
    "Task 19",
    "Task 20",
    "Task 21",
    "Task 22",
    "Task 23",
    "Task 24",
    "Task 25",
    "Task 26",
    "Task 27",
    "Task 28",
    "Task 29",
    "Task 30",
]


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Task

    @staticmethod
    def classify(prompt: str) -> Task:
        client = Anthropic()
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=64,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=Classification,
        )
        return response.parsed_output.task
