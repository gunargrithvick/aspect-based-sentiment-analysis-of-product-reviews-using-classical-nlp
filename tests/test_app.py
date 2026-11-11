"""Regression tests for the FastAPI browser application."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app import app


def request(method, path, **kwargs):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_homepage_serves_frontend() -> None:
    response = request("GET", "/")

    assert response.status_code == 200
    assert "Aspect-Based Sentiment Analyzer" in response.text
    assert "/static/css/styles.css" in response.text
    assert "Advanced review" in response.text
    assert "limitations-card" not in response.text


def test_health_reports_loaded_models() -> None:
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": True}


def test_analyze_endpoint_returns_structured_result() -> None:
    response = request("POST", "/api/analyze", json={"text": "The battery life is terrible."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == [
        {"aspect": "battery life", "sentiment": "negative", "start": 4, "end": 16}
    ]
    assert payload["summary"]["aspect_count"] == 1
    assert payload["summary"]["sentiment_counts"]["negative"] == 1


def test_analyze_endpoint_rejects_blank_review() -> None:
    response = request("POST", "/api/analyze", json={"text": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Review text must not be empty."


@pytest.mark.parametrize(
    ("text", "domain", "expected"),
    [
        ("The fridge is noisy.", "appliances", [("fridge", "negative")]),
        ("The earbuds sound crisp, but the microphone is muffled.", "audio",
         [("earbuds", "positive"), ("microphone", "negative")]),
        ("The phone works like a charm.", "phones", [("phone", "positive")]),
        ("My phone stopped working after one day.", "phones", [("phone", "negative")]),
        ("The vacuum has a cord.", "appliances", [("vacuum", "uncertain")]),
        ("The plot and acting were amazing.", None, []),
    ],
)
def test_product_domains_and_sentiment(text, domain, expected) -> None:
    response = request("POST", "/api/analyze", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert result["domains"] == ([domain] if domain else [])
    assert [(item["aspect"], item["sentiment"]) for item in result["results"]] == expected
    assert result["corrections"] == []
    assert result["possible_sarcasm"] is False


def test_spelling_correction_preserves_original_unicode_offsets() -> None:
    review = "\U0001f600 The batery is terrble."
    result = request("POST", "/api/analyze", json={"text": review}).json()
    assert result["review"] == review
    assert [(item["original"], item["replacement"]) for item in result["corrections"]] == [
        ("batery", "battery"), ("terrble", "terrible"),
    ]
    assert result["results"] == [{"aspect": "batery", "sentiment": "negative", "start": 6, "end": 12}]


def test_inferred_aspect_has_no_text_span() -> None:
    result = request("POST", "/api/analyze", json={"text": "It needs recharging every two hours."}).json()
    assert result["results"] == [
        {"aspect": "battery life", "sentiment": "negative", "start": None, "end": None, "implicit": True},
    ]


def test_sarcasm_flags_only_the_affected_clause() -> None:
    text = "Great, the phone died again. The camera is excellent."
    result = request("POST", "/api/analyze", json={"text": text}).json()
    assert result["possible_sarcasm"] is True
    assert result["summary"]["uncertain_count"] == 1
    assert [(item["aspect"], item["sentiment"]) for item in result["results"]] == [
        ("phone", "uncertain"), ("camera", "positive"),
    ]


def test_analyze_endpoint_limits_review_size() -> None:
    response = request("POST", "/api/analyze", json={"text": "a" * 5001})
    assert response.status_code == 422
