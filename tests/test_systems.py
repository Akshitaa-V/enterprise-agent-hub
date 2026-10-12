import sqlite3

import pytest

from agent_hub.systems.docs import DocumentStore
from agent_hub.systems.dwh import DWHError, Warehouse
from agent_hub.systems.erp import ERPClient, ERPError
from agent_hub.systems.legacy import RECORD_LENGTH, parse_export, parse_line


# ---------- legacy export ----------
def test_legacy_parser_reads_a_valid_record():
    row = parse_line("SKU-1003  Hinge, heavy duty       000045NUE220260927")
    assert (row.sku, row.qty, row.warehouse, row.updated) == ("SKU-1003", 45, "NUE2", "2026-09-27")


def test_legacy_parser_quarantines_bad_rows_instead_of_guessing():
    text = "\n".join([
        "# header",
        "SKU-1001  Steel bracket M8        001240MUC120260928",
        "SKU-1006  Anchor bolt set         00078XMUC120260928",  # non-numeric quantity
        "SKU-1009  Wall plug 8mm (100)     004410MUC1",          # truncated record
    ])
    result = parse_export(text)
    assert [r.sku for r in result.rows] == ["SKU-1001"]
    reasons = [q[2] for q in result.quarantined]
    assert "not numeric" in reasons[0] and f"expected {RECORD_LENGTH}" in reasons[1]


def test_legacy_load_keeps_quarantine_table(settings):
    con = sqlite3.connect(settings.erp_db)
    assert con.execute("SELECT COUNT(*) FROM stock").fetchone()[0] == 8
    assert con.execute("SELECT COUNT(*) FROM stock_quarantine").fetchone()[0] == 2


# ---------- ERP ----------
def test_invoice_status_is_derived_from_due_date_and_payments(settings):
    erp = ERPClient(settings)
    assert erp.get_invoice("INV-5015")["status"] == "overdue"
    assert erp.get_invoice("inv-5003")["status"] == "paid"  # ids are normalised
    assert erp.get_invoice("INV-5009")["status"] == "open"


def test_unknown_ids_raise_business_errors_with_hints(settings):
    erp = ERPClient(settings)
    with pytest.raises(ERPError) as exc:
        erp.get_order("SO-9999")
    assert "SO-1001" in exc.value.hint


def test_stock_for_quarantined_sku_explains_why(settings):
    with pytest.raises(ERPError, match="quarantined"):
        ERPClient(settings).get_stock("SKU-1006")


def test_ambiguous_customer_name_is_rejected(settings):
    with pytest.raises(ERPError, match="customers match"):
        ERPClient(settings).find_customer("e")


def test_credit_note_routes_to_the_right_approver(settings):
    erp = ERPClient(settings)
    assert erp.request_credit_note("INV-5004", 120, "short delivery")["approver_role"] == "account_manager"
    assert erp.request_credit_note("INV-5001", 800, "damaged goods")["approver_role"] == "head_of_finance"


def test_credit_note_cannot_exceed_invoice(settings):
    with pytest.raises(ERPError, match="exceeds invoice"):
        ERPClient(settings).request_credit_note("INV-5017", 5000, "pricing error")


def test_credit_note_approval_rules(settings):
    erp = ERPClient(settings)
    note = erp.request_credit_note("INV-5001", 800, "damaged goods")
    with pytest.raises(ERPError, match="head_of_finance"):
        erp.decide_credit_note(note["id"], approve=True, approver_role="account_manager")
    assert erp.decide_credit_note(note["id"], approve=True, approver_role="head_of_finance")["status"] == "booked"
    with pytest.raises(ERPError, match="already booked"):
        erp.decide_credit_note(note["id"], approve=False, approver_role="head_of_finance")


# ---------- data warehouse ----------
def test_metrics_match_raw_fact_table(settings):
    dwh = Warehouse(settings)
    total = sum(r["revenue"] for r in dwh.revenue_by_region(2025))
    raw = sqlite3.connect(settings.dwh_db).execute(
        "SELECT SUM(revenue) FROM fact_sales WHERE date_key LIKE '2025-%'").fetchone()[0]
    assert total == pytest.approx(raw, abs=0.05)


def test_future_and_missing_periods_are_rejected(settings):
    dwh = Warehouse(settings)
    with pytest.raises(DWHError, match="not happened"):
        dwh.revenue_by_region(2026, 4)
    with pytest.raises(DWHError, match="no data"):
        dwh.top_products(2019)


@pytest.mark.parametrize("sql", [
    "DELETE FROM fact_sales",
    "SELECT * FROM fact_sales; DROP TABLE fact_sales",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM customers",
    "PRAGMA table_info(fact_sales)",
])
def test_sql_guard_blocks_writes_and_unknown_tables(settings, sql):
    with pytest.raises(DWHError):
        Warehouse(settings).run_readonly_sql(sql)


def test_sql_guard_allows_a_join_over_the_star_schema(settings):
    rows = Warehouse(settings).run_readonly_sql(
        "SELECT p.category, COUNT(*) AS n FROM fact_sales f JOIN dim_product p ON p.sku = f.sku GROUP BY p.category")
    assert {r["category"] for r in rows} == {"Fasteners", "Hardware", "Electrical", "Lighting"}


# ---------- documents ----------
@pytest.mark.parametrize("query,source", [
    ("Who approves a credit note above EUR 500?", "credit_notes.md#approval-rules"),
    ("How long is the return window?", "returns_policy.md#return-window"),
    ("What does express delivery cost?", "shipping_sla.md#express-service"),
    ("early payment discount", "payment_terms.md#early-payment-discount"),
])
def test_document_search_ranks_the_right_section_first(settings, query, source):
    assert DocumentStore(settings.docs_dir).search(query)[0]["source"] == source


def test_document_search_returns_nothing_for_unrelated_queries(settings):
    assert DocumentStore(settings.docs_dir).search("quantum chromodynamics lattice") == []
