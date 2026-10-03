from fastapi.testclient import TestClient

from agent_hub import api
from agent_hub.graph import Hub


def client_for(settings) -> TestClient:
    hub = Hub(settings)
    api.app.dependency_overrides[api.get_hub] = lambda: hub
    return TestClient(api.app)


def test_health(settings):
    body = client_for(settings).get("/health").json()
    assert body["status"] == "ok" and body["llm_provider"] == "scripted"


def test_ask_then_fetch_trace(settings):
    client = client_for(settings)
    result = client.post("/ask", json={"question": "How many units of SKU-1003 are in stock?"}).json()
    assert "45 units" in result["answer"]
    run = client.get(f"/runs/{result['run_id']}").json()
    assert run["question"].startswith("How many units")
    assert client.get("/runs/doesnotexist").status_code == 404


def test_approval_flow_over_http(settings):
    client = client_for(settings)
    client.post("/ask", json={"question": "Create a credit note for INV-5004 of EUR 120 because of a short delivery"})
    pending = client.get("/approvals").json()
    assert len(pending) == 1 and pending[0]["approver_role"] == "account_manager"
    url = f"/approvals/{pending[0]['id']}/decision"
    assert client.post(url, json={"approve": True, "approver_role": "account_manager"}).json()["status"] == "booked"
    assert client.post(url, json={"approve": True, "approver_role": "account_manager"}).status_code == 409
    assert client.get("/approvals").json() == []


def test_input_validation(settings):
    assert client_for(settings).post("/ask", json={"question": ""}).status_code == 422
