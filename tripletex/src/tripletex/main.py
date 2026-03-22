from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import assert_never

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from tripletex.api_log_viewer import LogCollector, viewer_html
from tripletex.classifier import Classification
from tripletex.client import LoggedHTTPClient
from tripletex.my_log import (
    RequestContextLogMiddleware,
    configure_logging,
    unique_request_id_ctx,
)
from tripletex.pdf_parser import extract_text_from_pdf_base64
from tripletex.solve_request import InputFile, SolveRequest

logger = configure_logging()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(RequestContextLogMiddleware)  # ty:ignore[invalid-argument-type]


ATTACHMENT_TASKS = {
    "Task 19",
    "Task 20",
    "Task 21",
    "Task 22",
    "Task 23",
}

TRIPLETEX_PREFETCH_WORKERS = 15
TRIPLETEX_PREFETCH_ENDPOINTS = (
    "/employee",
    "/division",
    "/department",
    "/customer",
    # "/company",  # NOTE: doesn't work
    "/country",
    "/company/settings/altinn",
    "/contact",
    "/bank",
    "/bank/statement",
    "/deliveryAddress",
    "/employee/category",
    "/employee/employment",
    "/employee/employment/details",
    "/inventory",
    "/inventory/stocktaking",
    "/product",
    "/project",
    "/travelExpense",
    "/customer/category",
    "/employee/preferences",
    "/employee/entitlement",
    "/employee/standardTime",
    "/employee/nextOfKin",
    "/employee/employment/employmentType",
    "/employee/employment/workingHoursScheme",
    "/ledger/account",
    "/ledger/accountingPeriod",
    "/supplier",
    "ledger/vatType",
    "/salary/type",
    "/salary/payslip",
    "/travelExpense/costCategory",
    "/travelExpense/paymentType",
    "/ledger/accountingDimensionName",
    "/ledger/accountingDimensionValue",
)
TRIPLETEX_PREFETCH_ENDPOINTS_WITH_PARAMS = (
    ("/invoice", {"invoiceDateFrom": "1970-01-01", "invoiceDateTo": "2027-12-31"}),
    ("/balanceSheet", {"dateFrom": "1970-01-01", "dateTo": "2027-12-31"}),
    ("/inventory/inventories", {"dateFrom": "1970-01-01", "dateTo": "2027-12-31"}),
    ("/order", {"orderDateFrom": "1970-01-01", "orderDateTo": "2027-12-31"}),
    ("/ledger", {"dateFrom": "1970-01-01", "dateTo": "2027-12-31"}),
    ("/ledger/posting", {"dateFrom": "1970-01-01", "dateTo": "2027-12-31"}),
    ("/ledger/voucher", {"dateFrom": "1970-01-01", "dateTo": "2027-12-31"}),
    ("/travelExpense/rate", {"type": "PER_DIEM", "isValidDomestic": True}),
)


def attachment_to_text(input_file: InputFile) -> str:
    is_pdf = (
        input_file.mime_type == "application/pdf"
        or input_file.filename.lower().endswith(".pdf")
    )
    if is_pdf:
        return extract_text_from_pdf_base64(input_file.content_base64)

    try:
        normalized_content = "".join(input_file.content_base64.split())
        decoded_bytes = b64decode(normalized_content, validate=True)
    except BinasciiError as exc:
        msg = f"Attachment {input_file.filename!r} is not valid base64."
        raise ValueError(msg) from exc

    try:
        return decoded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"Attachment {input_file.filename!r} is not valid UTF-8 text."
        raise ValueError(msg) from exc


def get_task_attachment_text(*, task: str, files: list[InputFile]) -> str:
    if task not in ATTACHMENT_TASKS:
        msg = f"{task} does not accept attachments."
        raise ValueError(msg)

    if len(files) != 1:
        msg = f"{task} requires exactly one attachment, got {len(files)}."
        raise ValueError(msg)

    return attachment_to_text(files[0])


def get_submission_status(old_submission_status: str):  # TODO: remove this

    try:
        cookies = {
            "access_token": "",  # removed
        }

        logger.info("getting the submission status for nmai")
        response = requests.get(
            "https://api.ainm.no/tripletex/my/submissions", cookies=cookies
        )
        response.raise_for_status()

        logger.info(
            "got submission status for nmai", extra={"response": response.json()}
        )

        data = response.json()

        previous_submission_idx = [x["id"] for x in data].index(
            old_submission_status
        ) - 1
        current_request = data[previous_submission_idx]
        logger.info(
            f"got the most likely current run id {current_request['id']} (idx {previous_submission_idx})",
            extra={"data": current_request},
        )

    except Exception as e:
        logger.exception(
            "failed to get the submissions status for nmai",
            extra={"error": str(e)},
        )


def get_tripletex_data(
    *,
    endpoint: str,
    client_factory: Callable[[], LoggedHTTPClient],
    params: dict | None = None,
) -> None:
    try:
        logger.debug(
            "getting tripletex data",
            extra={"endpoint": endpoint, "params": params},
        )
        with client_factory() as client:
            response = client.get(endpoint, params=params)
        response.raise_for_status()
        logger.info(
            "got tripletex data",
            extra={"endpoint": endpoint, "params": params, "data": response.json()},
        )
    except Exception as e:
        logger.exception(
            "failed to get tripletex data",
            extra={"error": str(e), "endpoint": endpoint, "params": params},
        )


def prefetch_tripletex_data(*, client_factory: Callable[[], LoggedHTTPClient]) -> None:
    with ThreadPoolExecutor(
        max_workers=TRIPLETEX_PREFETCH_WORKERS,
        thread_name_prefix="tripletex-prefetch",
    ) as executor:
        futures = [
            executor.submit(
                copy_context().run,
                get_tripletex_data,
                endpoint=endpoint,
                client_factory=client_factory,
            )
            for endpoint in TRIPLETEX_PREFETCH_ENDPOINTS
        ]
        futures.extend(
            executor.submit(
                copy_context().run,
                get_tripletex_data,
                endpoint=endpoint,
                params=params,
                client_factory=client_factory,
            )
            for endpoint, params in TRIPLETEX_PREFETCH_ENDPOINTS_WITH_PARAMS
        )

        for future in futures:
            future.result()


@app.post("/solve")
def serve(payload: SolveRequest, request: Request) -> dict:
    # __import__("IPython").embed(header="")
    tripletex_client = payload.to_tripletex_client()
    previous_submission_id = request.headers["authorization"].split()[1]
    if previous_submission_id.startswith("DEBUG"):
        logger.debug("DEBUG RUN!!!")
        previous_submission_id = previous_submission_id[5:]
    logger.debug(
        "serve.handler.enter",
        extra={
            "prompt": payload.prompt,
            "file_count": len(payload.files),
            "base_url": tripletex_client.base_url,
            "payload": payload,
            "previous_submission_id": previous_submission_id,
        },
    )

    get_submission_status(previous_submission_id)  # TODO: add this

    collector = LogCollector(request_id=unique_request_id_ctx.get("-")).install()
    prefetch_tripletex_data(client_factory=payload.to_tripletex_client)

    logger.debug("serve.handler.going_to_main_run")

    task = Classification.classify(payload.prompt)

    logger.info(f"serve.handler.task {task}", extra={"task": task})

    match task:
        case "Task 1":
            from tripletex.parsers.task1_create_employee import CreateEmployee

            parsed = CreateEmployee.parse(payload.prompt)
        case "Task 2":
            from tripletex.parsers.task2_create_customer import CreateCustomer

            parsed = CreateCustomer.parse(payload.prompt)
        case "Task 3":
            from tripletex.parsers.task3_create_product import CreateProduct

            parsed = CreateProduct.parse(payload.prompt)
        case "Task 4":
            from tripletex.parsers.task4_create_supplier import CreateSupplier

            parsed = CreateSupplier.parse(payload.prompt)
        case "Task 5":
            from tripletex.parsers.task5_create_departments import CreateDepartments

            parsed = CreateDepartments.parse(payload.prompt)
        case "Task 6":
            from tripletex.parsers.task6_create_and_send_invoice import (
                CreateAndSendInvoice,
            )

            parsed = CreateAndSendInvoice.parse(payload.prompt)
        case "Task 7":
            from tripletex.parsers.task7_register_payment import RegisterPayment

            parsed = RegisterPayment.parse(payload.prompt)
        case "Task 8":
            from tripletex.parsers.task8_create_project import CreateProject

            parsed = CreateProject.parse(payload.prompt)
        case "Task 9":
            from tripletex.parsers.task9_multi_line_invoice import MultiLineInvoice

            parsed = MultiLineInvoice.parse(payload.prompt)
        case "Task 10":
            from tripletex.parsers.task10_order_to_invoice import OrderToInvoice

            parsed = OrderToInvoice.parse(payload.prompt)
        case "Task 11":
            from tripletex.parsers.task11_register_supplier_invoice import (
                RegisterSupplierInvoice,
            )

            parsed = RegisterSupplierInvoice.parse(payload.prompt)
        case "Task 12":
            from tripletex.parsers.task12_payroll_with_bonus import PayrollWithBonus

            parsed = PayrollWithBonus.parse(payload.prompt)
        case "Task 13":
            from tripletex.parsers.task13_travel_expense import TravelExpense

            parsed = TravelExpense.parse(payload.prompt)
        case "Task 14":
            from tripletex.parsers.task14_credit_note import CreditNote

            parsed = CreditNote.parse(payload.prompt)
        case "Task 15":
            from tripletex.parsers.task15_fixed_price_project import (
                FixedPriceProject,
            )

            parsed = FixedPriceProject.parse(payload.prompt)
        case "Task 16":
            from tripletex.parsers.task16_time_logging_invoice import (
                TimeLoggingInvoice,
            )

            parsed = TimeLoggingInvoice.parse(payload.prompt)
        case "Task 17":
            from tripletex.parsers.task17_custom_dimension import CustomDimension

            parsed = CustomDimension.parse(payload.prompt)
        case "Task 18":
            from tripletex.parsers.task18_reverse_payment import ReversePayment

            parsed = ReversePayment.parse(payload.prompt)
        case "Task 19":
            from tripletex.parsers.task19_employee_from_contract_pdf import (
                EmployeeFromContractPDF,
            )

            parsed = EmployeeFromContractPDF.parse(
                payload.prompt,
                get_task_attachment_text(task=task, files=payload.files),
            )
        case "Task 20":
            from tripletex.parsers.task20_supplier_invoice_pdf import (
                SupplierInvoiceFromPDF,
            )

            parsed = SupplierInvoiceFromPDF.parse(
                payload.prompt,
                get_task_attachment_text(task=task, files=payload.files),
            )
        case "Task 21":
            from tripletex.parsers.task21_onboarding_from_offer_pdf import (
                OnboardingFromOfferPDF,
            )

            parsed = OnboardingFromOfferPDF.parse(
                payload.prompt,
                get_task_attachment_text(task=task, files=payload.files),
            )
        case "Task 22":
            from tripletex.parsers.task22_expense_from_receipt_pdf import (
                ExpenseFromReceiptPDF,
            )

            parsed = ExpenseFromReceiptPDF.parse(
                payload.prompt,
                get_task_attachment_text(task=task, files=payload.files),
            )
        case "Task 23":
            from tripletex.parsers.task23_bank_reconciliation_csv import (
                BankReconciliation,
            )

            parsed = BankReconciliation.parse(
                payload.prompt,
                get_task_attachment_text(task=task, files=payload.files),
            )
        case "Task 24":
            from tripletex.parsers.task24_ledger_error_correction import (
                LedgerErrorCorrection,
            )

            parsed = LedgerErrorCorrection.parse(payload.prompt)
        case "Task 25":
            from tripletex.parsers.task25_overdue_reminder import OverdueReminder

            parsed = OverdueReminder.parse(payload.prompt)
        case "Task 26":
            from tripletex.parsers.task26_monthly_closing import MonthlyClosing

            parsed = MonthlyClosing.parse(payload.prompt)
        case "Task 27":
            from tripletex.parsers.task27_currency_invoice import CurrencyInvoice

            parsed = CurrencyInvoice.parse(payload.prompt)
        case "Task 28":
            from tripletex.parsers.task28_cost_increase_analysis import (
                CostIncreaseAnalysis,
            )

            parsed = CostIncreaseAnalysis.parse(payload.prompt)
        case "Task 29":
            from tripletex.parsers.task29_complete_project_lifecycle import (
                CompleteProjectLifecycle,
            )

            parsed = CompleteProjectLifecycle.parse(payload.prompt)
        case "Task 30":
            from tripletex.parsers.task30_year_end_closing import YearEndClosing

            parsed = YearEndClosing.parse(payload.prompt)
        case _:
            assert_never(task)

    logger.info("serve.handler.parsed_task", extra={"task": task, "parsed": parsed})
    try:
        parsed.solve(tripletex_client)
    except Exception as e:
        logger.exception(
            "failed to run solve",
            extra={"error": str(e)},
        )

    collector.uninstall()
    collector.save_to_disk(
        prompt=payload.prompt,
        files=[f.filename for f in payload.files],
        request_id=unique_request_id_ctx.get("-"),
    )

    logger.debug("serve.handler.before_return")
    return {"status": "completed"}


@app.get("/logs", response_class=HTMLResponse)
def logs_viewer(request: Request) -> str:
    """Browse all logged /solve requests and their API call timelines."""
    return viewer_html(request_id=request.query_params.get("request"))


if __name__ == "__main__":
    uvicorn.run(
        "tripletex.main:app",
        host="0.0.0.0",
        port=8000,
        access_log=False,
        log_config=None,
        reload=False,
    )
