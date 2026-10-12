"""Builds the demo ERP and data warehouse databases from a fixed random seed.

The data is fictional but shaped like a mid-sized B2B distributor: customers,
sales orders, invoices and a star-schema sales warehouse. Running the seed twice
produces byte-identical data, so tests and evals are reproducible.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

from .config import Settings, get_settings

CUSTOMERS = [
    ("C-01", "Brennwerk GmbH", "North", "Lena Hoffmann"),
    ("C-02", "Nordlicht Logistik", "North", "Lena Hoffmann"),
    ("C-03", "Alpenmetall AG", "South", "Jonas Weber"),
    ("C-04", "Isarblick Medien", "South", "Jonas Weber"),
    ("C-05", "Donau Foods", "East", "Mira Schulz"),
    ("C-06", "Kraemer Elektro", "East", "Mira Schulz"),
    ("C-07", "Lechtal Bau", "West", "Tim Becker"),
    ("C-08", "Sonnenhof Hotels", "West", "Tim Becker"),
]

PRODUCTS = [
    ("SKU-1001", "Steel bracket M8", "Fasteners", 4.20),
    ("SKU-1002", "Steel bracket M10", "Fasteners", 5.10),
    ("SKU-1003", "Hinge, heavy duty", "Hardware", 18.50),
    ("SKU-1004", "Cable duct 2m", "Electrical", 9.80),
    ("SKU-1005", "Mounting rail 1m", "Hardware", 12.40),
    ("SKU-1006", "Anchor bolt set", "Fasteners", 22.00),
    ("SKU-1007", "Junction box IP65", "Electrical", 31.90),
    ("SKU-1008", "LED panel 60x60", "Lighting", 74.00),
    ("SKU-1009", "Wall plug 8mm (100)", "Fasteners", 6.60),
    ("SKU-1010", "Safety switch 3-pole", "Electrical", 129.00),
]

TODAY = date(2026, 10, 1)  # fixed "today" so overdue logic is reproducible


def _connect(path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return sqlite3.connect(path)


def seed_erp(settings: Settings) -> None:
    rng = random.Random(7)
    con = _connect(settings.erp_db)
    con.executescript(
        """
        CREATE TABLE customers (id TEXT PRIMARY KEY, name TEXT NOT NULL, region TEXT, account_manager TEXT,
                                credit_limit REAL NOT NULL);
        CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers(id),
                             order_date TEXT, status TEXT, total REAL);
        CREATE TABLE invoices (id TEXT PRIMARY KEY, order_id TEXT REFERENCES orders(id), customer_id TEXT,
                               invoice_date TEXT, due_date TEXT, amount REAL, paid REAL);
        CREATE TABLE stock (sku TEXT PRIMARY KEY, description TEXT, qty INTEGER, warehouse TEXT, updated TEXT);
        CREATE TABLE credit_notes (id TEXT PRIMARY KEY, invoice_id TEXT, amount REAL, reason TEXT,
                                   status TEXT, approver_role TEXT, created_at TEXT, decided_at TEXT);
        """
    )
    con.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)",
        [(cid, name, region, am, 10000.0 if i % 3 else 25000.0) for i, (cid, name, region, am) in enumerate(CUSTOMERS)],
    )
    statuses = ["delivered", "delivered", "delivered", "shipped", "in_production", "confirmed"]
    for n in range(20):
        oid = f"SO-{1001 + n}"
        cust = CUSTOMERS[n % len(CUSTOMERS)][0]
        odate = TODAY - timedelta(days=rng.randint(5, 80))
        total = round(rng.uniform(300, 9000), 2)
        status = rng.choice(statuses)
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (oid, cust, odate.isoformat(), status, total))
        if status in ("delivered", "shipped"):
            inv_date = odate + timedelta(days=2)
            due = inv_date + timedelta(days=30)
            paid_roll = rng.random()
            paid = total if paid_roll < 0.5 else (round(total * 0.4, 2) if paid_roll < 0.6 else 0.0)
            con.execute(
                "INSERT INTO invoices VALUES (?,?,?,?,?,?,?)",
                (f"INV-{5001 + n}", oid, cust, inv_date.isoformat(), due.isoformat(), total, paid),
            )
    con.commit()
    con.close()


def seed_dwh(settings: Settings) -> None:
    rng = random.Random(11)
    con = _connect(settings.dwh_db)
    con.executescript(
        """
        CREATE TABLE dim_customer (customer_id TEXT PRIMARY KEY, name TEXT, region TEXT);
        CREATE TABLE dim_product (sku TEXT PRIMARY KEY, name TEXT, category TEXT, unit_price REAL);
        CREATE TABLE dim_date (date_key TEXT PRIMARY KEY, year INTEGER, quarter INTEGER, month INTEGER);
        CREATE TABLE fact_sales (date_key TEXT REFERENCES dim_date(date_key), customer_id TEXT, sku TEXT,
                                 qty INTEGER, revenue REAL);
        """
    )
    con.executemany("INSERT INTO dim_customer VALUES (?,?,?)", [(c, n, r) for c, n, r, _ in CUSTOMERS])
    con.executemany("INSERT INTO dim_product VALUES (?,?,?,?)", PRODUCTS)
    d = date(2025, 1, 1)
    while d <= date(2026, 9, 30):
        con.execute(
            "INSERT INTO dim_date VALUES (?,?,?,?)", (d.isoformat(), d.year, (d.month - 1) // 3 + 1, d.month)
        )
        if d.weekday() < 5:
            growth = 1.0 + 0.012 * ((d.year - 2025) * 12 + d.month - 1)  # slow upward trend
            for _ in range(rng.randint(1, 4)):
                cust = rng.choice(CUSTOMERS)[0]
                sku, _, _, price = rng.choice(PRODUCTS)
                qty = max(1, int(rng.gauss(40, 15) * growth))
                con.execute(
                    "INSERT INTO fact_sales VALUES (?,?,?,?,?)", (d.isoformat(), cust, sku, qty, round(qty * price, 2))
                )
        d += timedelta(days=1)
    con.commit()
    con.close()


def seed_all(settings: Settings | None = None) -> None:
    from .systems.legacy import load_legacy_inventory

    settings = settings or get_settings()
    seed_erp(settings)
    seed_dwh(settings)
    load_legacy_inventory(settings)


if __name__ == "__main__":
    seed_all()
    print("Seeded ERP, data warehouse and legacy inventory into", get_settings().var_dir)
