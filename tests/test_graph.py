"""End-to-end tests of the multi-agent graph, including models that misbehave."""

import json
from dataclasses import replace

from langchain_core.messages import AIMessage

from agent_hub.graph import Hub
from agent_hub.llm import ScriptedChatModel
from agent_hub.tools import Services, erp_tools


def test_single_agent_question(settings):
    result = Hub(settings).ask("What is the status of order SO-1004?")
    assert result["metrics"]["agents"] == ["erp"]
    assert "delivered" in result["answer"] and "INV-5004" in result["answer"]


def test_three_agents_answer_one_question(settings):
    result = Hub(settings).ask("For Brennwerk GmbH: list overdue invoices, revenue in 2025, and our policy on late payment.")
    assert result["metrics"]["agents"] == ["erp", "analytics", "docs"]
    for fact in ("INV-5001", "INV-5017", "EUR 110,295.30", "[payment_terms.md#late-payment]"):
        assert fact in result["answer"]


def test_errors_are_reported_not_invented(settings):
    result = Hub(settings).ask("What is the status of order SO-9999?")
    assert "does not exist" in result["answer"]
    assert result["metrics"]["tool_errors"] == 1


def test_credit_note_waits_for_approval(settings):
    hub = Hub(settings)
    result = hub.ask("Create a credit note for INV-5001 of EUR 800 because of damaged goods")
    assert len(result["approvals"]) == 1
    assert result["approvals"][0]["status"] == "pending_approval"
    assert hub.services.erp.list_credit_notes("booked") == []


def test_run_trace_is_persisted(settings):
    hub = Hub(settings)
    result = hub.ask("Which invoices are overdue for Lechtal Bau, and how much revenue did Lechtal Bau bring in 2025?")
    run = hub.runs.get(result["run_id"])
    kinds = [e["kind"] for e in run["events"]]
    assert kinds.count("route") == 3  # erp, analytics, finish
    assert any(e["kind"] == "tool" and e["name"] == "customer_revenue" for e in run["events"])


# ---------- guards against a misbehaving model ----------
class RepeatingSupervisor(ScriptedChatModel):
    """Always routes the same task to the ERP agent."""

    def _supervise(self, block):
        return AIMessage(content="", tool_calls=[{"name": "route_to_agent", "id": "r", "type": "tool_call",
                                                   "args": {"next_agent": "erp", "task": "status of SO-1001"}}])


class WanderingSupervisor(ScriptedChatModel):
    """Never finishes: a new task every hop."""

    def _supervise(self, block):
        n = block.count("- [")
        return AIMessage(content="", tool_calls=[{"name": "route_to_agent", "id": "r", "type": "tool_call",
                                                   "args": {"next_agent": "docs", "task": f"policy question {n}"}}])


class ToolHungryModel(ScriptedChatModel):
    """Supervisor sends work to ERP once; the ERP agent then never stops calling tools."""

    def _respond(self, messages):
        if "route_to_agent" in self.bound_tool_names:
            done = "COMPLETED: none" not in messages[-1].content
            nxt = "finish" if done else "erp"
            return AIMessage(content="", tool_calls=[{"name": "route_to_agent", "id": "r", "type": "tool_call",
                                                       "args": {"next_agent": nxt, "task": "loop"}}])
        if not self.bound_tool_names:
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[{"name": "get_order", "id": "c", "type": "tool_call",
                                                   "args": {"order_id": "SO-1001"}}])


class ForbiddenToolModel(ToolHungryModel):
    """The ERP agent tries to call a warehouse tool it has no access to, then a malformed id."""

    def _respond(self, messages):
        if self.bound_tool_names and "route_to_agent" not in self.bound_tool_names and len(messages) == 2:
            return AIMessage(content="", tool_calls=[
                {"name": "run_readonly_sql", "id": "c1", "type": "tool_call", "args": {"sql": "SELECT 1"}},
                {"name": "get_order", "id": "c2", "type": "tool_call", "args": {"order_id": "1001; DROP"}},
            ])
        if self.bound_tool_names and "route_to_agent" not in self.bound_tool_names:
            return AIMessage(content="could not complete")
        return super()._respond(messages)


def test_loop_detection_stops_repeated_routing(settings):
    result = Hub(settings, llm=RepeatingSupervisor()).ask("anything")
    assert result["metrics"]["stop_reason"] == "loop_blocked"
    assert result["metrics"]["handoffs"] == 1
    assert "stopped early" in result["answer"]


def test_hop_limit_stops_a_supervisor_that_never_finishes(settings):
    s = replace(settings, max_hops=3)
    result = Hub(s, llm=WanderingSupervisor()).ask("anything")
    assert result["metrics"]["stop_reason"] == "max_hops"
    assert result["metrics"]["handoffs"] == 3


def test_tool_call_limit_per_agent(settings):
    result = Hub(settings, llm=ToolHungryModel()).ask("anything")
    assert "tool_call_limit" in result["metrics"]["guard_stops"]
    assert result["metrics"]["tool_calls"] == settings.max_tool_calls_per_agent + 1


def test_agents_cannot_use_other_agents_tools_or_pass_bad_arguments(settings):
    hub = Hub(settings, llm=ForbiddenToolModel())
    result = hub.ask("anything")
    tool_events = [e for e in hub.runs.get(result["run_id"])["events"] if e["kind"] == "tool"]
    assert [e["ok"] for e in tool_events] == [False, False]
    assert "not available to the erp agent" in tool_events[0]["error"]
    assert "invalid call" in tool_events[1]["error"]


def test_tool_envelope_on_validation_error(settings):
    get_invoice = {t.name: t for t in erp_tools(Services.from_settings(settings))}["get_invoice"]
    ok = json.loads(get_invoice.invoke({"invoice_id": "INV-5001"}))
    missing = json.loads(get_invoice.invoke({"invoice_id": "INV-0001"}))
    assert ok["ok"] and ok["data"]["status"] == "overdue"
    assert not missing["ok"] and "INV-5001" in missing["hint"]
