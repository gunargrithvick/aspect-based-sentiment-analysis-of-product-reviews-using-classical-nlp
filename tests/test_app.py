"""Regression tests for the FastAPI browser application."""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_homepage_serves_frontend() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Aspect-Based Sentiment Analyzer" in response.text
    assert "/static/css/styles.css" in response.text


def test_health_reports_loaded_models() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": True}


def test_analyze_endpoint_returns_structured_result() -> None:
    response = client.post("/api/analyze", json={"text": "The battery life is terrible."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == [
        {"aspect": "battery life", "sentiment": "negative", "start": 4, "end": 16}
    ]
    assert payload["summary"]["aspect_count"] == 1
    assert payload["summary"]["sentiment_counts"]["negative"] == 1


def test_analyze_endpoint_rejects_blank_review() -> None:
    response = client.post("/api/analyze", json={"text": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Review text must not be empty."
