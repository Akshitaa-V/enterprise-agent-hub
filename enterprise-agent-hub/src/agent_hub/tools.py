"""LangChain tools for each specialist agent.

Every tool returns JSON with the same envelope:
  {"ok": true, "data": ...}                       on success
  {"ok": false, "error": "...", "hint": "..."}    on a business or validation error
Errors are returned instead of raised so the model can read them and recover
(fix an id, narrow a name) instead of the whole run crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .config import Settings
from .systems.docs import DocumentStore
from .systems.dwh import DWHError, Warehouse
from .systems.erp import ERPClient, ERPError


@dataclass
class Services:
    settings: Settings
    erp: ERPClient
    dwh: Warehouse
    docs: DocumentStore

    @classmethod
    def from_settings(cls, settings: Settings) -> "Services":
        return cls(settings, ERPClient(settings), Warehouse(settings), DocumentStore(settings.docs_dir))


def _envelope(fn: Callable) -> Callable:
    def wrapper(**kwargs):
        try:
            return json.dumps({"ok": True, "data": fn(**kwargs)}, default=str)
        except (ERPError, DWHError) as exc:
            return json.dumps({"ok": False, "error": str(exc), "hint": exc.hint})
    return wrapper


# ---------- argument schemas (validated before any system is called) ----------
class OrderArgs(BaseModel):
    order_id: str = Field(description="Sales order id, e.g. SO-1004", pattern=r"(?i)^SO-\d{4}$")


class InvoiceArgs(BaseModel):
    invoice_id: str = Field(description="Invoice id, e.g. INV-5004", pattern=r"(?i)^INV-\d{4}$")


class OverdueArgs(BaseModel):
    customer_name: Optional[str] = Field(default=None, description="Customer name or part of it; omit for all customers")


class StockArgs(BaseModel):
    sku: str = Field(description="Article number, e.g. SKU-1003", pattern=r"(?i)^SKU-\d{4}$")


class CreditNoteArgs(BaseModel):
    invoice_id: str = Field(pattern=r"(?i)^INV-\d{4}$")
    amount: float = Field(gt=0, description="Credit amount in EUR")
    reason: str = Field(min_length=3, description="Why the credit note is needed")


class RegionArgs(BaseModel):
    year: int
    quarter: Optional[int] = Field(default=None, description="1 to 4; omit for the full year")


class TopProductsArgs(BaseModel):
    year: int
    limit: int = Field(default=3, ge=1, le=10)


class YearArgs(BaseModel):
    year: int


class CustomerRevenueArgs(BaseModel):
    customer_name: str
    year: int


class SQLArgs(BaseModel):
    sql: str = Field(description="One SELECT over fact_sales, dim_customer, dim_product, dim_date")


class SearchArgs(BaseModel):
    query: str = Field(min_length=3)


def _tool(name: str, description: str, schema: type[BaseModel], fn: Callable) -> BaseTool:
    return StructuredTool.from_function(func=_envelope(fn), name=name, description=description, args_schema=schema)


def erp_tools(s: Services) -> list[BaseTool]:
    return [
        _tool("get_order", "Look up a sales order: status, total, customer and linked invoice.", OrderArgs, s.erp.get_order),
        _tool("get_invoice", "Look up an invoice: amount, paid, open amount, due date and overdue status.", InvoiceArgs,
              s.erp.get_invoice),
        _tool("list_overdue_invoices", "List overdue invoices, optionally for one customer.", OverdueArgs,
              lambda customer_name=None: s.erp.overdue_invoices(customer_name)),
        _tool("get_stock_level", "Current stock for an article, loaded from the legacy warehouse export.", StockArgs,
              s.erp.get_stock),
        _tool("request_credit_note",
              "Create a credit note REQUEST. It is not booked until a human approves it.", CreditNoteArgs,
              s.erp.request_credit_note),
    ]


def analytics_tools(s: Services) -> list[BaseTool]:
    return [
        _tool("revenue_by_region", "Revenue per sales region for a year or one quarter.", RegionArgs, s.dwh.revenue_by_region),
        _tool("top_products", "Best-selling products by revenue in a year.", TopProductsArgs, s.dwh.top_products),
        _tool("quarterly_revenue", "Total revenue per quarter for a year.", YearArgs, s.dwh.quarterly_revenue),
        _tool("customer_revenue", "Revenue from one customer in a year.", CustomerRevenueArgs, s.dwh.customer_revenue),
        _tool("run_readonly_sql", "Ad-hoc read-only SQL when no metric fits. Max 200 rows.", SQLArgs,
              s.dwh.run_readonly_sql),
    ]


def docs_tools(s: Services) -> list[BaseTool]:
    return [
        _tool("search_documents", "Search company policies (payment terms, returns, credit notes, shipping, onboarding).",
              SearchArgs, lambda query: s.docs.search(query, k=3)),
    ]
