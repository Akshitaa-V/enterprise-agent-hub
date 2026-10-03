"""Parser for the fixed-width inventory export of the legacy warehouse system.

The export is Latin-1 encoded with CRLF line endings and fixed column widths.
Bad rows are not guessed at: they are quarantined with a reason, and only
valid rows are loaded into the ERP stock table.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings

COLUMNS = [("sku", 10), ("description", 24), ("qty", 6), ("warehouse", 4), ("updated", 8)]
RECORD_LENGTH = sum(width for _, width in COLUMNS)


@dataclass
class StockRow:
    sku: str
    description: str
    qty: int
    warehouse: str
    updated: str  # ISO date


@dataclass
class ParseResult:
    rows: list[StockRow]
    quarantined: list[tuple[int, str, str]]  # (line number, raw line, reason)


def parse_line(raw: str) -> StockRow:
    if len(raw) != RECORD_LENGTH:
        raise ValueError(f"expected {RECORD_LENGTH} characters, got {len(raw)}")
    fields, pos = {}, 0
    for name, width in COLUMNS:
        fields[name] = raw[pos : pos + width]
        pos += width
    if not fields["qty"].isdigit():
        raise ValueError(f"quantity {fields['qty']!r} is not numeric")
    try:
        updated = datetime.strptime(fields["updated"], "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"invalid date {fields['updated']!r}") from exc
    return StockRow(
        sku=fields["sku"].strip(),
        description=fields["description"].strip(),
        qty=int(fields["qty"]),
        warehouse=fields["warehouse"].strip(),
        updated=updated,
    )


def parse_export(text: str) -> ParseResult:
    rows, quarantined = [], []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        try:
            rows.append(parse_line(raw))
        except ValueError as exc:
            quarantined.append((number, raw, str(exc)))
    return ParseResult(rows, quarantined)


def load_legacy_inventory(settings: Settings) -> ParseResult:
    text = settings.legacy_export.read_bytes().decode("latin-1")
    result = parse_export(text)
    con = sqlite3.connect(settings.erp_db)
    con.execute("DELETE FROM stock")
    con.executemany(
        "INSERT INTO stock VALUES (?,?,?,?,?)",
        [(r.sku, r.description, r.qty, r.warehouse, r.updated) for r in result.rows],
    )
    con.execute("CREATE TABLE IF NOT EXISTS stock_quarantine (line INTEGER, raw TEXT, reason TEXT)")
    con.execute("DELETE FROM stock_quarantine")
    con.executemany("INSERT INTO stock_quarantine VALUES (?,?,?)", result.quarantined)
    con.commit()
    con.close()
    return result
