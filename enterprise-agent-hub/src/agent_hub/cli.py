"""Command line: ask one question and print the answer with its trace summary.

    python -m agent_hub.cli "Which invoices are overdue for Brennwerk GmbH?"
    python -m agent_hub.cli --trace "..."   # also print every routing decision and tool call
"""

from __future__ import annotations

import argparse
import json

from .config import get_settings
from .graph import Hub
from .seed import seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the enterprise agent hub a question.")
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="print the full trace")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.erp_db.exists():
        seed_all(settings)
    result = Hub(settings).ask(args.question)
    print(result["answer"])
    m = result["metrics"]
    print(f"\nrun {result['run_id']} | agents: {', '.join(m['agents']) or 'none'} | tool calls: {m['tool_calls']} "
          f"| tool errors: {m['tool_errors']} | {m['latency_ms']} ms")
    if args.trace:
        print(json.dumps(Hub(settings).runs.get(result["run_id"])["events"], indent=2))


if __name__ == "__main__":
    main()
