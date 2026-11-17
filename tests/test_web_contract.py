"""HTTP fields the web lane can rely on for assistant round reviews."""
from test_assist_routes import H, env  # noqa: F401


def test_review_contract(env, monkeypatch):  # noqa: F811 - imported pytest fixture
    client, _, _ = env
    monkeypatch.setenv("KALOS_ASSIST_ASK_PER_MINUTE", "0")
    monkeypatch.setattr("kalos.assist.routes.AssistIndex.search", lambda *a, **kw: [])
    response = client.post("/api/assist/review", headers=H("a-read"), json={})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"claims", "report", "abstained", "ledger", "next"}
    assert body["claims"] == [] and body["abstained"] is True
    assert isinstance(body["report"], str) and isinstance(body["next"], dict)
    assert all(set(e) == {"role", "kind", "claim_id", "verdict", "reasons"}
               for e in body["ledger"])
    roles = client.get("/api/assist/review/roles", headers=H("a-read"))
    assert roles.status_code == 200
    assert all(set(role) == {"key", "model_group", "model", "instructions"}
               for role in roles.json())
    assert client.get("/api/assist/review/roles").status_code == 401
    assert client.post("/api/assist/review", json={}).status_code == 401
