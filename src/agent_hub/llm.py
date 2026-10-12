"""Model factory, plus a deterministic offline model for tests and CI.

`ScriptedChatModel` is a real LangChain chat model (it supports bind_tools and
returns tool calls), but it decides with keyword rules instead of a neural
network. It lets the whole graph, every tool and the eval harness run without
API keys and with identical results on every run. Set HUB_LLM_PROVIDER=openai
or anthropic to run the same graph with a hosted model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .config import Settings

ERP_WORDS = re.compile(
    r"\b(SO-\d{4}|INV-\d{4}|SKU-\d{4}|overdue|outstanding|unpaid|in stock|stock level|status of order)\b", re.I
)
ANALYTICS_WORDS = re.compile(
    r"\b(revenue|sales|best[- ]selling|top \d+|top products?|by region|per region|quarterly|by quarter)\b", re.I
)
DOCS_WORDS = re.compile(
    r"\b(policy|policies|allowed|who approves|who must approve|approval rules?|payment terms|how long|how many days|"
    r"return|restocking|shipping|delivery time|express|response time|onboarding|VAT|discount|rules?)\b",
    re.I,
)
AMOUNT = re.compile(r"(?:EUR|€)\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(?:EUR|€|euros?)", re.I)
NAME_AFTER = re.compile(r"\b(?i:for|from|of|by)\s+([A-Z][\w&]*(?:\s+[A-Z][\w&]*)*)")


def _eur(x: float) -> str:
    return f"EUR {x:,.2f}"


def _amount(text: str) -> float | None:
    m = AMOUNT.search(text)
    if not m:
        return None
    return float((m.group(1) or m.group(2)).replace(",", ""))


CAPITALISED_NAME = re.compile(r"(?<!^)(?<![.?!:] )\b([A-Z][a-z]+(?:\s+[A-Z][\w&]+)+)\b")
AGENT_PATTERNS = (("erp", ERP_WORDS), ("analytics", ANALYTICS_WORDS), ("docs", DOCS_WORDS))


def _name(text: str) -> str | None:
    for m in NAME_AFTER.finditer(text):
        candidate = m.group(1)
        if not re.match(r"(?i)^(Q\d|EUR|SO-|INV-|SKU-)", candidate):
            return candidate
    m = CAPITALISED_NAME.search(text)  # "did Lechtal Bau bring", no preposition before the name
    return m.group(1) if m else None


def _subtask(question: str, agent: str) -> str:
    """Give each specialist only the clauses meant for it, plus clauses no agent claims
    (shared context such as "For Brennwerk GmbH")."""
    clauses = [c.strip() for c in re.split(r",\s+(?:and\s+)?|;|:\s|\band\b", question) if c.strip()]
    if len(clauses) < 2:
        return question
    rx = dict(AGENT_PATTERNS)[agent]
    mine = []
    for i, c in enumerate(clauses):
        if rx.search(c):
            # "who must approve it": a pronoun needs the clause it refers to, minus ids the agent cannot use
            if i and re.search(r"\b(it|this|that|them)\b", c, re.I):
                mine.append(re.sub(r"\b(?:SO|INV|SKU)-\d{4}\b", "", clauses[i - 1]))
            mine.append(c)
    shared = [c for c in clauses if not any(r.search(c) for _, r in AGENT_PATTERNS)]
    return "; ".join(shared + mine) if mine else question


class ScriptedChatModel(BaseChatModel):
    bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedChatModel":
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"bound_tool_names": names})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    # ---------- behaviour ----------
    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        human = next(m.content for m in messages if isinstance(m, HumanMessage))
        tools = set(self.bound_tool_names)
        if "route_to_agent" in tools:
            return self._supervise(human)
        if not tools:
            return self._synthesize(human)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=self._summarize([m for m in messages if isinstance(m, ToolMessage)]))
        task = human.split("TASK:", 1)[-1].strip()
        calls = self._plan_calls(task, tools)
        if not calls:
            return AIMessage(content="I could not map this task to any of my tools.")
        return AIMessage(
            content="",
            tool_calls=[{"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)],
        )

    def _supervise(self, block: str) -> AIMessage:
        question = re.search(r"QUESTION: (.*)", block).group(1)
        done = set(re.search(r"COMPLETED: (.*)", block).group(1).split(", ")) - {"none"}
        needed = [a for a, rx in AGENT_PATTERNS if rx.search(question)] or ["docs"]
        nxt = next((a for a in needed if a not in done), "finish")
        task = "" if nxt == "finish" else _subtask(question, nxt)
        return AIMessage(content="", tool_calls=[{"name": "route_to_agent", "id": "route_0", "type": "tool_call",
                                                   "args": {"next_agent": nxt, "task": task}}])

    @staticmethod
    def _synthesize(block: str) -> AIMessage:
        findings = re.findall(r"^- \[\w+\] (.*)$", block, flags=re.M)
        return AIMessage(content=" ".join(findings) if findings else "I could not find an answer in the connected systems.")

    @staticmethod
    def _plan_calls(task: str, tools: set[str]) -> list[tuple[str, dict]]:
        calls: list[tuple[str, dict]] = []
        if "get_order" in tools:  # ERP specialist
            inv = re.findall(r"INV-\d{4}", task, re.I)
            if re.search(r"credit note", task, re.I) and inv and re.search(r"\b(create|request|issue|raise)\b", task, re.I):
                reason = re.search(r"\b(?:because|for|due to)\s+(.*?)[.?]?$", task)
                calls.append(("request_credit_note", {"invoice_id": inv[0], "amount": _amount(task) or 0.0,
                                                      "reason": reason.group(1) if reason else "requested by user"}))
                return calls
            calls += [("get_order", {"order_id": o}) for o in re.findall(r"SO-\d{4}", task, re.I)]
            calls += [("get_invoice", {"invoice_id": i}) for i in inv]
            calls += [("get_stock_level", {"sku": s}) for s in re.findall(r"SKU-\d{4}", task, re.I)]
            if re.search(r"\b(overdue|outstanding|unpaid)\b", task, re.I) and not inv:
                name = _name(task)
                calls.append(("list_overdue_invoices", {"customer_name": name} if name else {}))
        elif "revenue_by_region" in tools:  # analytics specialist
            years = re.findall(r"\b(20\d\d)\b", task)
            year = int(years[0]) if years else 2026
            quarter = re.search(r"\bQ([1-4])\b", task, re.I)
            if re.search(r"region", task, re.I):
                args = {"year": year} | ({"quarter": int(quarter.group(1))} if quarter else {})
                calls.append(("revenue_by_region", args))
            elif re.search(r"\b(top|best)", task, re.I):
                n = re.search(r"top (\d+)", task, re.I)
                calls.append(("top_products", {"year": year, "limit": int(n.group(1)) if n else 3}))
            elif re.search(r"quarter", task, re.I):
                calls.append(("quarterly_revenue", {"year": year}))
            elif _name(task):
                calls.append(("customer_revenue", {"customer_name": _name(task), "year": year}))
            else:
                calls.append(("quarterly_revenue", {"year": year}))
        elif "search_documents" in tools:
            calls.append(("search_documents", {"query": task}))
        return calls

    @staticmethod
    def _summarize(results: list[ToolMessage]) -> str:
        parts = []
        for msg in results:
            payload = json.loads(msg.content)
            if not payload["ok"]:
                hint = f" ({payload['hint']})" if payload.get("hint") else ""
                parts.append(f"{msg.name} failed: {payload['error']}{hint}.")
                continue
            parts.append(describe(msg.name, payload["data"]))
        return " ".join(parts)


def describe(tool: str, d: Any) -> str:
    if tool == "get_order":
        inv = f"invoice {d['invoice_id']}" if d["invoice_id"] else "no invoice yet"
        return (f"Order {d['id']} for {d['customer']} from {d['order_date']} is {d['status'].replace('_', ' ')}, "
                f"total {_eur(d['total'])}, {inv}.")
    if tool == "get_invoice":
        late = f" ({d['days_overdue']} days)" if d["status"] == "overdue" else ""
        return (f"Invoice {d['id']} for {d['customer']}: amount {_eur(d['amount'])}, paid {_eur(d['paid'])}, "
                f"open {_eur(d['open_amount'])}, due {d['due_date']}, status {d['status']}{late}.")
    if tool == "list_overdue_invoices":
        if not d:
            return "No overdue invoices found."
        total = sum(i["open_amount"] for i in d)
        items = "; ".join(f"{i['id']} {i['customer']} {_eur(i['open_amount'])} ({i['days_overdue']} days)" for i in d)
        return f"{len(d)} overdue invoice(s), {_eur(total)} open in total: {items}."
    if tool == "get_stock_level":
        return f"{d['sku']} ({d['description']}): {d['qty']} units in warehouse {d['warehouse']}, updated {d['updated']}."
    if tool == "request_credit_note":
        who = d["approver_role"].replace("_", " ")
        return (f"Credit note request {d['id']} for {d['invoice_id']} over {_eur(d['amount'])} was created and is "
                f"pending approval by the {who}; it is not booked yet.")
    if tool == "revenue_by_region":
        return "Revenue by region: " + ", ".join(f"{r['region']} {_eur(r['revenue'])}" for r in d) + "."
    if tool == "top_products":
        return "Top products by revenue: " + "; ".join(
            f"{n}. {r['name']} ({r['sku']}) {_eur(r['revenue'])}" for n, r in enumerate(d, 1)) + "."
    if tool == "quarterly_revenue":
        return "Revenue by quarter: " + ", ".join(f"Q{r['quarter']} {_eur(r['revenue'])}" for r in d) + "."
    if tool == "customer_revenue":
        return f"Revenue from {d['name']}: {_eur(d['revenue'])}."
    if tool == "run_readonly_sql":
        return f"The query returned {len(d)} row(s): {json.dumps(d[:5])}."
    if tool == "search_documents":
        if not d:
            return "No policy document covers this."
        top = d[:1] + [h for h in d[1:2] if h["score"] >= 0.9 * d[0]["score"]]
        return " ".join(f"{h['text']} [{h['source']}]" for h in top)
    return json.dumps(d)


def get_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "scripted":
        return ScriptedChatModel()
    if not settings.llm_model:
        raise ValueError("set HUB_LLM_MODEL to the model name of your provider")
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.llm_model, temperature=0)
    raise ValueError(f"unknown HUB_LLM_PROVIDER {settings.llm_provider!r}")
