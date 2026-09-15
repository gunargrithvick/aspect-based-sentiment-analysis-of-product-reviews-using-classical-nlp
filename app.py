"""FastAPI entry point for the local classical ABSA application."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.inference_pipeline import AspectBasedSentimentAnalyzer

app = FastAPI(
    title="Classical Aspect-Based Sentiment Analyzer",
    description="A local FastAPI frontend for the classical NLP ABSA system.",
    version="1.0.0",
)
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "public" / "static"),
    name="static",
)
templates = Jinja2Templates(directory=PROJECT_ROOT / "frontend" / "templates")


class ReviewRequest(BaseModel):
    """Request body for one product review."""

    text: str = Field(..., min_length=1, description="English product review text")


@lru_cache(maxsize=1)
def get_analyzer() -> AspectBasedSentimentAnalyzer:
    """Load the trained models once and reuse them for local requests."""

    return AspectBasedSentimentAnalyzer.from_model_directory(PROJECT_ROOT / "models")


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Serve the browser interface."""

    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    """Report whether the trained NLP models are available."""

    try:
        get_analyzer()
    except FileNotFoundError:
        return {"status": "error", "models_loaded": False}
    return {"status": "ok", "models_loaded": True}


@app.post("/api/analyze")
def analyze_review(payload: ReviewRequest) -> dict:
    """Analyze a product review with the trained classical NLP pipeline."""

    review_text = payload.text.strip()
    if not review_text:
        raise HTTPException(status_code=400, detail="Review text must not be empty.")

    try:
        results = get_analyzer().analyze(review_text)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    sentiment_labels = ("positive", "negative", "neutral", "conflict")
    summary = {
        "aspect_count": len(results),
        "sentiment_counts": {
            label: sum(result["sentiment"] == label for result in results)
            for label in sentiment_labels
        },
    }
    return {"review": review_text, "results": results, "summary": summary}
