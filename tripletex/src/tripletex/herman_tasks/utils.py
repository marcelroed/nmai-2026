from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal, Self
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel, model_validator

from tripletex.client import LoggedHTTPClient


class TripletexCredentials(BaseModel):
    base_url: str
    auth: tuple[Literal[0], str]

    @model_validator(mode="before")
    @classmethod
    def coerce_session_token(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        if "session_token" not in data:
            return data

        if "auth" in data:
            raise ValueError("Provide either 'auth' or 'session_token', not both.")

        data = dict(data)
        session_token = data.pop("session_token")
        data["auth"] = (0, session_token)
        return data

    @classmethod
    def placeholder_TODO(cls) -> Self:
        from tripletex.config import AUTH, BASE_URL

        return cls(base_url=BASE_URL, auth=(0, AUTH[1]))

    def to_client(self) -> LoggedHTTPClient:
        return LoggedHTTPClient(base_url=self.base_url, auth=("0", self.auth[1]))


TripletexAPI = TripletexCredentials | LoggedHTTPClient


def _get(api: TripletexAPI, endpoint: str, **kwargs: Any) -> requests.Response:
    if isinstance(api, LoggedHTTPClient):
        return api.get(endpoint, **kwargs)
    return requests.get(f"{api.base_url}{endpoint}", auth=api.auth, **kwargs)


def get_current_year_month_day_utc(days_offset_forward: int = 0) -> str:
    current_date_utc = datetime.now(ZoneInfo("UTC")).date()
    return (current_date_utc + timedelta(days=days_offset_forward)).strftime("%Y-%m-%d")


def get_department_by_name(
    department_name: str, tripletex_client: TripletexAPI
) -> dict:
    departments = _get(tripletex_client, "/department").json()

    (department,) = [
        x for x in departments["values"] if x["name"].lower() == department_name.lower()
    ]
    return department


def get_customer_by_org_number(
    org_number: str, tripletex_client: TripletexAPI
) -> dict:
    customers = _get(tripletex_client, "/customer").json()

    (customer,) = [
        x for x in customers["values"] if x["organizationNumber"] == org_number
    ]
    return customer


def get_product_by_product_number(
    product_number: str, tripletex_client: TripletexAPI
) -> dict:
    products = _get(tripletex_client, "/product").json()

    (product,) = [
        product for product in products["values"] if product["number"] == product_number
    ]
    return product


def get_products_by_product_numbers(
    product_numbers: list[str], tripletex_client: TripletexAPI
) -> list[dict]:
    return [
        get_product_by_product_number(
            product_number=product_number, tripletex_client=tripletex_client
        )
        for product_number in product_numbers
    ]


def get_supplier_by_org_num(
    org_num: str, tripletex_client: TripletexAPI
) -> dict:
    suppliers = _get(tripletex_client, "/supplier").json()

    (supplier,) = [x for x in suppliers["values"] if x["organizationNumber"] == org_num]
    return supplier


def get_nok_currency(tripletex_client: TripletexAPI) -> dict:
    (nok_currency,) = [
        x
        for x in _get(tripletex_client, "/currency").json()["values"]
        if x["displayName"] == "NOK"
    ]
    return nok_currency


def get_employee(employee_email: str, tripletex_client: TripletexAPI) -> dict:
    employees = _get(tripletex_client, "/employee").json()
    (empolyee,) = [x for x in employees["values"] if x["email"] == employee_email]
    return empolyee


def get_invoice_by_amount_excluding_vat(
    amount_excluding_vat: float, tripletex_client: TripletexAPI
) -> dict:
    invoices = _get(
        tripletex_client,
        "/invoice",
        params={"invoiceDateFrom": "1970-01-01", "invoiceDateTo": "2027-12-31"},
    ).json()
    (invoice,) = [
        x for x in invoices["values"] if x["amountExcludingVat"] == amount_excluding_vat
    ]
    return invoice
