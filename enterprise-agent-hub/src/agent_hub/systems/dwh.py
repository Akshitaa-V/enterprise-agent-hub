"""Read-only access to the sales data warehouse (star schema).

Agents get two kinds of access: named metrics with fixed SQL, which are safe and
fast, and a guarded read-only SQL tool for questions the metrics do not cover.
The guard allows a single SELECT over the warehouse tables and nothing else.
"""

from __future__ import annotations

import re
import sqlite3

from ..config import Settings

ALLOWED_TABLES = {"fact_sales", "dim_customer", "dim_product", "dim_date"}
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.IGNORECASE
)
MAX_ROWS = 200


class DWHError(Exception):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class Warehouse:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        # mode=ro: even if the guard missed something, SQLite refuses writes.
        con = sqlite3.connect(f"file:{self.settings.dwh_db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(sql, params).fetchmany(MAX_ROWS)]
        finally:
            con.close()

    # ---------- named metrics ----------
    def revenue_by_region(self, year: int, quarter: int | None = None) -> list[dict]:
        self._check_period(year, quarter)
        sql = """SELECT c.region, ROUND(SUM(f.revenue), 2) AS revenue
                 FROM fact_sales f JOIN dim_customer c USING (customer_id) JOIN dim_date d USING (date_key)
                 WHERE d.year = ?"""
        params: tuple = (year,)
        if quarter:
            sql += " AND d.quarter = ?"
            params += (quarter,)
        return self._query(sql + " GROUP BY c.region ORDER BY revenue DESC", params)

    def top_products(self, year: int, limit: int = 3) -> list[dict]:
        self._check_period(year, None)
        return self._query(
            """SELECT p.sku, p.name, ROUND(SUM(f.revenue), 2) AS revenue
               FROM fact_sales f JOIN dim_product p USING (sku) JOIN dim_date d USING (date_key)
               WHERE d.year = ? GROUP BY p.sku ORDER BY revenue DESC LIMIT ?""",
            (year, max(1, min(limit, 10))),
        )

    def quarterly_revenue(self, year: int) -> list[dict]:
        self._check_period(year, None)
        return self._query(
            """SELECT d.quarter, ROUND(SUM(f.revenue), 2) AS revenue
               FROM fact_sales f JOIN dim_date d USING (date_key) WHERE d.year = ?
               GROUP BY d.quarter ORDER BY d.quarter""",
            (year,),
        )

    def customer_revenue(self, customer_name: str, year: int) -> dict:
        self._check_period(year, None)
        rows = self._query(
            """SELECT c.name, ROUND(SUM(f.revenue), 2) AS revenue
               FROM fact_sales f JOIN dim_customer c USING (customer_id) JOIN dim_date d USING (date_key)
               WHERE d.year = ? AND lower(c.name) LIKE ? GROUP BY c.name""",
            (year, f"%{customer_name.strip().lower()}%"),
        )
        if len(rows) != 1:
            raise DWHError(f"{len(rows)} customers match {customer_name!r} in the warehouse")
        return rows[0]

    @staticmethod
    def _check_period(year: int, quarter: int | None) -> None:
        if year not in (2025, 2026):
            raise DWHError(f"no data for {year}", "the warehouse holds Jan 2025 to Sep 2026")
        if quarter is not None and quarter not in (1, 2, 3, 4):
            raise DWHError(f"quarter must be 1 to 4, got {quarter}")
        if year == 2026 and quarter == 4:
            raise DWHError("Q4 2026 has not happened yet", "the warehouse holds Jan 2025 to Sep 2026")

    # ---------- guarded ad-hoc SQL ----------
    @staticmethod
    def validate_sql(sql: str) -> str:
        cleaned = sql.strip().rstrip(";").strip()
        if ";" in cleaned:
            raise DWHError("only one statement is allowed")
        if not re.match(r"(?is)^(select|with)\b", cleaned):
            raise DWHError("only SELECT queries are allowed")
        if FORBIDDEN.search(cleaned):
            raise DWHError("the query contains a forbidden keyword")
        tables = {t.lower() for t in re.findall(r"(?i)\b(?:from|join)\s+([a-z_][a-z0-9_]*)", cleaned)}
        unknown = tables - ALLOWED_TABLES
        if unknown:
            raise DWHError(f"unknown tables: {', '.join(sorted(unknown))}", f"allowed: {', '.join(sorted(ALLOWED_TABLES))}")
        return cleaned

    def run_readonly_sql(self, sql: str) -> list[dict]:
        return self._query(self.validate_sql(sql))
