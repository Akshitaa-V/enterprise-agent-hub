"""The multi-agent graph: a supervisor routes work to ERP, analytics and documents
specialists, each running its own tool-calling loop, and a synthesizer writes
the final answer from their findings.

    START -> supervisor -> (erp | analytics | docs) -> supervisor -> ... -> synthesize -> END

Guards: a hop limit, a per-agent tool-call limit, loop detection (the same task
sent to the same agent twice), tool allow-lists per agent, and argument
validation before any enterprise system is called.
"""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph

from . import prompts
from .config import Settings, get_settings
from .llm import get_llm
from .tools import Services, analytics_tools, docs_tools, erp_tools
from .tracing import RunStore, Tracer

SPECIALISTS = ("erp", "analytics", "docs")


class HubState(TypedDict, total=False):
    question: str
    findings: Annotated[list[dict], operator.add]
    approvals: Annotated[list[dict], operator.add]
    routed: Annotated[list[str], operator.add]
    hops: int
    next_agent: str
    task: str
    stop_reason: str
    answer: str


@tool
def route_to_agent(next_agent: Literal["erp", "analytics", "docs", "finish"], task: str = "") -> str:
    """Send a self-contained task to one specialist, or finish when the question is fully answered."""
    return next_agent


def _tracer(config: RunnableConfig) -> Tracer:
    return config["configurable"]["tracer"]


def build_graph(settings: Settings, services: Services, llm: BaseChatModel | None = None):
    llm = llm or get_llm(settings)
    supervisor_llm = llm.bind_tools([route_to_agent])

    def supervisor(state: HubState, config: RunnableConfig) -> dict:
        tracer, hops = _tracer(config), state.get("hops", 0)
        if hops >= settings.max_hops:
            tracer.event("guard", "supervisor", reason="max_hops")
            return {"next_agent": "finish", "stop_reason": "max_hops"}
        ai = supervisor_llm.invoke([SystemMessage(prompts.SUPERVISOR),
                                    HumanMessage(prompts.status_block(state["question"], state.get("findings", [])))])
        tracer.add_usage(ai)
        call = ai.tool_calls[0] if ai.tool_calls else {"args": {"next_agent": "finish", "task": ""}}
        nxt, task = call["args"].get("next_agent", "finish"), (call["args"].get("task") or state["question"]).strip()
        if nxt != "finish" and nxt not in SPECIALISTS:
            tracer.event("guard", "supervisor", reason=f"invalid_route:{nxt}")
            return {"next_agent": "finish", "stop_reason": "invalid_route"}
        key = f"{nxt}::{task.lower()}"
        if nxt != "finish" and key in state.get("routed", []):
            tracer.event("guard", "supervisor", reason="loop_blocked", repeated=nxt)
            return {"next_agent": "finish", "stop_reason": "loop_blocked"}
        tracer.event("route", "supervisor", next_agent=nxt, task=task)
        update: dict = {"next_agent": nxt, "task": task, "hops": hops + 1}
        if nxt != "finish":
            update["routed"] = [key]
        return update

    def make_specialist(name: str, system_prompt: str, tools: list[BaseTool]):
        bound = llm.bind_tools(tools)
        by_name = {t.name: t for t in tools}

        def run(state: HubState, config: RunnableConfig) -> dict:
            tracer = _tracer(config)
            messages = [SystemMessage(system_prompt),
                        HumanMessage(f"ORIGINAL QUESTION: {state['question']}\nTASK: {state['task']}")]
            approvals, text = [], ""
            for _ in range(settings.max_tool_calls_per_agent + 1):
                ai = bound.invoke(messages, config)
                tracer.add_usage(ai)
                messages.append(ai)
                if not ai.tool_calls:
                    text = ai.content if isinstance(ai.content, str) else json.dumps(ai.content)
                    break
                for call in ai.tool_calls:
                    result = _call_tool(by_name, call, tracer, name)
                    if call["name"] == "request_credit_note" and json.loads(result)["ok"]:
                        approvals.append(json.loads(result)["data"])
                    messages.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
            else:
                tracer.event("guard", name, reason="tool_call_limit")
                text = f"Stopped after {settings.max_tool_calls_per_agent} tool rounds without a final result."
            return {"findings": [{"agent": name, "task": state["task"], "text": text.strip()}], "approvals": approvals}

        return run

    synth_llm = llm  # no tools: the synthesizer can only use what the specialists found

    def synthesize(state: HubState, config: RunnableConfig) -> dict:
        tracer = _tracer(config)
        ai = synth_llm.invoke([SystemMessage(prompts.SYNTHESIZER),
                               HumanMessage(prompts.status_block(state["question"], state.get("findings", [])))])
        tracer.add_usage(ai)
        answer = ai.content if isinstance(ai.content, str) else json.dumps(ai.content)
        if state.get("stop_reason") in ("max_hops", "loop_blocked"):
            answer += " (Note: the agents stopped early, so this answer may be incomplete.)"
        tracer.event("synthesize", "synthesizer", chars=len(answer))
        return {"answer": answer}

    graph = StateGraph(HubState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("erp", make_specialist("erp", prompts.ERP, erp_tools(services)))
    graph.add_node("analytics", make_specialist("analytics", prompts.ANALYTICS, analytics_tools(services)))
    graph.add_node("docs", make_specialist("docs", prompts.DOCS, docs_tools(services)))
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", lambda s: "synthesize" if s["next_agent"] == "finish" else s["next_agent"],
        ["erp", "analytics", "docs", "synthesize"],
    )
    for name in SPECIALISTS:
        graph.add_edge(name, "supervisor")
    graph.add_edge("synthesize", END)
    return graph.compile()


def _call_tool(by_name: dict[str, BaseTool], call: dict, tracer: Tracer, agent: str) -> str:
    start = time.perf_counter()
    if call["name"] not in by_name:  # the model asked for a tool this agent is not allowed to use
        result = json.dumps({"ok": False, "error": f"tool {call['name']} is not available to the {agent} agent",
                             "hint": f"available: {', '.join(by_name)}"})
    else:
        try:
            result = by_name[call["name"]].invoke(call["args"])
        except Exception as exc:  # argument validation errors and unexpected failures
            first_line = str(exc).splitlines()[0]
            result = json.dumps({"ok": False, "error": f"invalid call: {first_line}", "hint": "check the arguments"})
    payload = json.loads(result)
    tracer.event("tool", agent, name=call["name"], args=call["args"], ok=payload["ok"],
                 error=payload.get("error"), latency_ms=round((time.perf_counter() - start) * 1000, 2))
    return result


class Hub:
    """Entry point used by the API, the CLI and the eval harness."""

    def __init__(self, settings: Settings | None = None, llm: BaseChatModel | None = None):
        self.settings = settings or get_settings()
        self.services = Services.from_settings(self.settings)
        self.graph = build_graph(self.settings, self.services, llm)
        self.runs = RunStore(self.settings.runs_db)

    def ask(self, question: str) -> dict:
        tracer = Tracer()
        callbacks = []
        if self.settings.langfuse_enabled:
            from langfuse.langchain import CallbackHandler

            callbacks.append(CallbackHandler())
        state = self.graph.invoke(
            {"question": question, "hops": 0},
            config={"configurable": {"tracer": tracer}, "callbacks": callbacks, "recursion_limit": 40,
                    "run_name": "enterprise-agent-hub", "metadata": {"run_id": tracer.run_id}},
        )
        metrics = tracer.metrics()
        metrics["stop_reason"] = state.get("stop_reason", "answered")
        self.runs.save(tracer, question, state["answer"], metrics)
        return {"run_id": tracer.run_id, "answer": state["answer"], "findings": state.get("findings", []),
                "approvals": state.get("approvals", []), "metrics": metrics}
