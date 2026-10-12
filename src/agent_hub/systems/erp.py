"""ERP adapter.

In a customer project this class wraps the ERP's REST API. Here it runs against a
SQLite simulator with the same contract, so agents, tests and evals do not need
a live ERP. Read methods never change data; the only write path (credit notes)
creates a request that a human has to approve before it is booked.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import date, datetime

from ..config import Settings
from ..seed import TODAY


class ERPError(Exception):
    """A business-level error the agent can explain or recover from."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class ERPClient:
    def __init__(self, settings: Settings, today: date = TODAY):
        self.settings = settings
        self.today = today

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.settings.erp_db)
        con.row_factory = sqlite3.Row
        return con

    # ---------- reads ----------
    def get_order(self, order_id: str) -> dict:
        order_id = order_id.strip().upper()
        with self._con() as con:
            row = con.execute(
                "SELECT o.*, c.name AS customer FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = ?",
                (order_id,),
            ).fetchone()
            if row is None:
                raise ERPError(f"order {order_id} does not exist", "order ids look like SO-1001")
            invoice = con.execute("SELECT id FROM invoices WHERE order_id = ?", (order_id,)).fetchone()
        result = dict(row)
        result["invoice_id"] = invoice["id"] if invoice else None
        return result

    def get_invoice(self, invoice_id: str) -> dict:
        invoice_id = invoice_id.strip().upper()
        with self._con() as con:
            row = con.execute(
                "SELECT i.*, c.name AS customer FROM invoices i JOIN customers c ON c.id = i.customer_id WHERE i.id = ?",
                (invoice_id,),
            ).fetchone()
        if row is None:
            raise ERPError(f"invoice {invoice_id} does not exist", "invoice ids look like INV-5001")
        return self._with_status(dict(row))

    def _with_status(self, inv: dict) -> dict:
        open_amount = round(inv["amount"] - inv["paid"], 2)
        due = date.fromisoformat(inv["due_date"])
        inv["open_amount"] = open_amount
        if open_amount <= 0:
            inv["status"], inv["days_overdue"] = "paid", 0
        elif due < self.today:
            inv["status"], inv["days_overdue"] = "overdue", (self.today - due).days
        else:
            inv["status"], inv["days_overdue"] = "open", 0
        return inv

    def find_customer(self, name: str) -> dict:
        with self._con() as con:
            rows = con.execute(
                "SELECT * FROM customers WHERE lower(name) LIKE ?", (f"%{name.strip().lower()}%",)
            ).fetchall()
        if not rows:
            raise ERPError(f"no customer matches {name!r}")
        if len(rows) > 1:
            raise ERPError(f"{len(rows)} customers match {name!r}", "use a more specific name")
        return dict(rows[0])

    def overdue_invoices(self, customer_name: str | None = None) -> list[dict]:
        sql = "SELECT i.*, c.name AS customer FROM invoices i JOIN customers c ON c.id = i.customer_id"
        params: tuple = ()
        if customer_name:
            customer = self.find_customer(customer_name)
            sql += " WHERE i.customer_id = ?"
            params = (customer["id"],)
        with self._con() as con:
            rows = [self._with_status(dict(r)) for r in con.execute(sql + " ORDER BY i.id", params)]
        return [r for r in rows if r["status"] == "overdue"]

    def get_stock(self, sku: str) -> dict:
        sku = sku.strip().upper()
        with self._con() as con:
            row = con.execute("SELECT * FROM stock WHERE sku = ?", (sku,)).fetchone()
            if row is None:
                bad = con.execute("SELECT reason FROM stock_quarantine WHERE raw LIKE ?", (sku + "%",)).fetchone()
        if row is None:
            if bad:
                raise ERPError(
                    f"stock for {sku} is unknown: the legacy export row was quarantined ({bad['reason']})",
                    "ask the warehouse team to re-export this SKU",
                )
            raise ERPError(f"SKU {sku} does not exist", "SKUs look like SKU-1001")
        return dict(row)

    # ---------- writes that need approval ----------
    def request_credit_note(self, invoice_id: str, amount: float, reason: str) -> dict:
        inv = self.get_invoice(invoice_id)
        if amount <= 0:
            raise ERPError("credit note amount must be positive")
        if amount > inv["amount"]:
            raise ERPError(
                f"credit note of EUR {amount:,.2f} exceeds invoice {inv['id']} amount of EUR {inv['amount']:,.2f}",
                "a credit note can never exceed the original invoice",
            )
        role = "head_of_finance" if amount > self.settings.finance_approval_threshold else "account_manager"
        note = {
            "id": f"CN-{uuid.uuid4().hex[:8].upper()}",
            "invoice_id": inv["id"],
            "amount": round(amount, 2),
            "reason": reason,
            "status": "pending_approval",
            "approver_role": role,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decided_at": None,
        }
        with self._con() as con:
            con.execute("INSERT INTO credit_notes VALUES (:id,:invoice_id,:amount,:reason,:status,"
                        ":approver_role,:created_at,:decided_at)", note)
        return note

    def list_credit_notes(self, status: str | None = None) -> list[dict]:
        with self._con() as con:
            if status:
                rows = con.execute("SELECT * FROM credit_notes WHERE status = ?", (status,)).fetchall()
            else:
                rows = con.execute("SELECT * FROM credit_notes").fetchall()
        return [dict(r) for r in rows]

    def decide_credit_note(self, note_id: str, approve: bool, approver_role: str) -> dict:
        with self._con() as con:
            row = con.execute("SELECT * FROM credit_notes WHERE id = ?", (note_id,)).fetchone()
            if row is None:
                raise ERPError(f"credit note {note_id} does not exist")
            if row["status"] != "pending_approval":
                raise ERPError(f"credit note {note_id} is already {row['status']}")
            if approve and approver_role != row["approver_role"] and approver_role != "head_of_finance":
                raise ERPError(f"credit note {note_id} needs approval by {row['approver_role']}")
            status = "booked" if approve else "rejected"
            con.execute(
                "UPDATE credit_notes SET status = ?, decided_at = ? WHERE id = ?",
                (status, datetime.now().isoformat(timespec="seconds"), note_id),
            )
        return {**dict(row), "status": status}
