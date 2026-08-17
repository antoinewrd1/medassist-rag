from fastapi.testclient import TestClient

from app.analyze import NO_CONTEXT_SUMMARY
from app.main import app


def test_health_reports_backends():
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["documents"] > 0


def test_corpus_exposes_provenance():
    with TestClient(app) as client:
        body = client.get("/corpus").json()
        assert all("Synthetic" in d["provenance"] for d in body["documents"])


def test_ordinary_symptoms_get_a_grounded_cited_answer():
    with TestClient(app) as client:
        body = client.post("/analyze", json={"symptoms": "burning when I urinate"}).json()
        assert body["grounded"] is True
        assert body["citations"]
        assert body["llm_invoked"] is True


def test_emergency_short_circuits_before_the_model():
    with TestClient(app) as client:
        body = client.post(
            "/analyze", json={"symptoms": "crushing chest pain radiating to my jaw"}
        ).json()
        assert body["triage"] == "emergency"
        assert body["llm_invoked"] is False       # the model was never consulted
        assert "911" in body["directive"]


def test_self_harm_returns_crisis_resource():
    with TestClient(app) as client:
        body = client.post(
            "/analyze", json={"symptoms": "I don't want to live anymore"}
        ).json()
        assert body["crisis_resource"] is True
        assert "988" in body["directive"]


def test_out_of_domain_refuses_rather_than_guessing():
    with TestClient(app) as client:
        body = client.post(
            "/analyze", json={"symptoms": "how do I fix my car transmission"}
        ).json()
        assert body["summary"] == NO_CONTEXT_SUMMARY
        assert body["citations"] == []


def test_every_response_carries_the_disclaimer():
    with TestClient(app) as client:
        for symptoms in ["mild cough", "chest pain", "how to fix a carburetor"]:
            body = client.post("/analyze", json={"symptoms": symptoms}).json()
            assert "not a medical device" in body["disclaimer"]


def test_short_input_is_rejected_at_the_boundary():
    with TestClient(app) as client:
        assert client.post("/analyze", json={"symptoms": "x"}).status_code == 422
