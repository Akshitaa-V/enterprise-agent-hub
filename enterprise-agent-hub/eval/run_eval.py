"""Runs the labelled eval set against the full graph and reports routing accuracy,
answer accuracy, approvals, tool errors and latency.

    python eval/run_eval.py                      # offline, deterministic
    HUB_LLM_PROVIDER=openai HUB_LLM_MODEL=... python eval/run_eval.py

Each run starts from freshly seeded data in a temporary directory, so results
do not depend on earlier runs (credit notes created by one case cannot leak into another).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="hub-eval-")
    os.environ["HUB_VAR_DIR"] = tmp
    from agent_hub.config import get_settings
    from agent_hub.graph import Hub
    from agent_hub.seed import seed_all

    settings = get_settings()
    seed_all(settings)
    hub = Hub(settings)
    cases = [json.loads(line) for line in (ROOT / "eval" / "cases.jsonl").read_text().splitlines() if line.strip()]

    rows = []
    for case in cases:
        result = hub.ask(case["question"])
        answer, m = result["answer"], result["metrics"]
        missing = [s for s in case["must_contain"] if s.lower() not in answer.lower()]
        forbidden = [s for s in case["must_not_contain"] if s.lower() in answer.lower()]
        routing_ok = set(m["agents"]) == set(case["expected_agents"])
        answer_ok = not missing and not forbidden
        approvals_ok = len(result["approvals"]) == case["expected_approvals"]
        rows.append({"id": case["id"], "category": case["category"], "routing_ok": routing_ok, "answer_ok": answer_ok,
                     "approvals_ok": approvals_ok, "passed": routing_ok and answer_ok and approvals_ok,
                     "agents": m["agents"], "missing": missing, "forbidden": forbidden,
                     "tool_calls": m["tool_calls"], "tool_errors": m["tool_errors"], "latency_ms": m["latency_ms"],
                     "tokens": m["input_tokens"] + m["output_tokens"], "answer": answer})

    n = len(rows)
    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "provider": settings.llm_provider,
        "model": settings.llm_model or "scripted",
        "cases": n,
        "passed": sum(r["passed"] for r in rows),
        "routing_accuracy": round(sum(r["routing_ok"] for r in rows) / n, 3),
        "answer_accuracy": round(sum(r["answer_ok"] for r in rows) / n, 3),
        "approval_accuracy": round(sum(r["approvals_ok"] for r in rows) / n, 3),
        "multi_agent_cases_passed": f"{sum(r['passed'] for r in rows if r['category'] == 'multi')}"
                                    f"/{sum(r['category'] == 'multi' for r in rows)}",
        "failure_cases_passed": f"{sum(r['passed'] for r in rows if r['category'] == 'failure')}"
                                f"/{sum(r['category'] == 'failure' for r in rows)}",
        "tool_calls": sum(r["tool_calls"] for r in rows),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "tokens": sum(r["tokens"] for r in rows),
    }
    out = ROOT / "eval" / f"results_{settings.llm_provider}.json"
    out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))

    for r in rows:
        flag = "PASS" if r["passed"] else "FAIL"
        detail = "" if r["passed"] else f"  agents={r['agents']} missing={r['missing']} forbidden={r['forbidden']}"
        print(f"{flag}  {r['id']:<9} {r['latency_ms']:>7.1f} ms{detail}")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
