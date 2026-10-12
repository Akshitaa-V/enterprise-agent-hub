"""Per-run trace of every routing decision, tool call and model call.

Langfuse (when keys are set) receives the full LangChain trace. This local trace
is always on: it powers the run API, the metrics the eval harness reports, and
debugging without any external service.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Tracer:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    events: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def event(self, kind: str, agent: str, **data) -> None:
        self.events.append({"t_ms": round((time.perf_counter() - self.started) * 1000, 1), "kind": kind,
                            "agent": agent, **data})

    def add_usage(self, message) -> None:
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def metrics(self) -> dict:
        tools = [e for e in self.events if e["kind"] == "tool"]
        routes = [e for e in self.events if e["kind"] == "route" and e.get("next_agent") != "finish"]
        return {
            "latency_ms": round((time.perf_counter() - self.started) * 1000, 1),
            "agents": [e["next_agent"] for e in routes],
            "handoffs": len(routes),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for e in tools if not e["ok"]),
            "guard_stops": [e["reason"] for e in self.events if e["kind"] == "guard"],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class RunStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT, question TEXT, "
                        "answer TEXT, metrics TEXT, events TEXT)")

    def save(self, tracer: Tracer, question: str, answer: str, metrics: dict) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                        (tracer.run_id, datetime.now().isoformat(timespec="seconds"), question, answer,
                         json.dumps(metrics), json.dumps(tracer.events, default=str)))

    def get(self, run_id: str) -> dict | None:
        with sqlite3.connect(self.path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["metrics"], out["events"] = json.loads(out["metrics"]), json.loads(out["events"])
        return out
