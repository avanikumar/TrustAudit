import math

from fastapi.testclient import TestClient


def test_audit_endpoint(monkeypatch):
    monkeypatch.setenv("TRUSTAUDIT_DATASET", "cancer")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from trustaudit.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["writer"] == "TemplateWriter"
        example = client.get("/schema").json()["example"]
        res = client.post("/audit", json={"features": example})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] in {"verified", "partially_verified", "failed"}
        assert not math.isnan(body["probability"])

        bad = client.post("/audit", json={"features": {"x": 1}})
        assert bad.status_code == 422
