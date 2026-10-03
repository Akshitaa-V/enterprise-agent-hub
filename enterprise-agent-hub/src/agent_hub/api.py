"""HTTP API.

POST /ask                              ask a question; returns the answer, findings, approvals and metrics
GET  /runs/{run_id}                    full trace of a run (routing decisions, tool calls, latencies)
GET  /approvals                        credit note requests waiting for a human
POST /approvals/{note_id}/decision     approve or reject a credit note request
GET  /health                           liveness and configuration
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .graph import Hub
from .systems.erp import ERPError

app = FastAPI(title="Enterprise Agent Hub", version=__version__)


@lru_cache
def get_hub() -> Hub:
    return Hub()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Decision(BaseModel):
    approve: bool
    approver_role: Literal["account_manager", "head_of_finance"]


@app.get("/health")
def health(hub: Hub = Depends(get_hub)) -> dict:
    return {"status": "ok", "version": __version__, "llm_provider": hub.settings.llm_provider,
            "langfuse": hub.settings.langfuse_enabled}


@app.post("/ask")
def ask(req: AskRequest, hub: Hub = Depends(get_hub)) -> dict:
    return hub.ask(req.question)


@app.get("/runs/{run_id}")
def get_run(run_id: str, hub: Hub = Depends(get_hub)) -> dict:
    run = hub.runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run


@app.get("/approvals")
def pending_approvals(hub: Hub = Depends(get_hub)) -> list[dict]:
    return hub.services.erp.list_credit_notes(status="pending_approval")


@app.post("/approvals/{note_id}/decision")
def decide(note_id: str, decision: Decision, hub: Hub = Depends(get_hub)) -> dict:
    try:
        return hub.services.erp.decide_credit_note(note_id, decision.approve, decision.approver_role)
    except ERPError as exc:
        raise HTTPException(409, str(exc)) from exc
