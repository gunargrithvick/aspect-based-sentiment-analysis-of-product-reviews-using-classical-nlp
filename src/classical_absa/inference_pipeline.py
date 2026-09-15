"""Standalone inference pipeline for aspect-based sentiment analysis."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


class AspectBasedSentimentAnalyzer:
    """Combine the trained CRF extractor and SVM sentiment classifier."""

    def __init__(self, aspect_extractor, sentiment_classifier) -> None:
        self.aspect_extractor = aspect_extractor
        self.sentiment_classifier = sentiment_classifier

    @classmethod
    def from_model_directory(cls, model_directory: str | Path) -> "AspectBasedSentimentAnalyzer":
        """Load the default improved models from a model directory."""

        directory = Path(model_directory)
        extractor_path = directory / "crf_aspect_extractor.joblib"
        classifier_path = directory / "svm_sentiment_classifier.joblib"
        missing = [str(path) for path in (extractor_path, classifier_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required trained model files are missing: "
                + ", ".join(missing)
                + ". Run scripts/train_improved_models.py first."
            )
        return cls(joblib.load(extractor_path), joblib.load(classifier_path))

    def analyze(self, text: str) -> list[dict]:
        """Return aspect, sentiment, and character-offset results for one review."""

        if not text or not text.strip():
            raise ValueError("Review text must not be empty.")

        aspects = self.aspect_extractor.predict(text)
        if not aspects:
            return []

        aspect_frame = pd.DataFrame(
            [{"sentence_text": text, **aspect} for aspect in aspects]
        )
        polarities = self.sentiment_classifier.predict(aspect_frame)
        return [
            {
                "aspect": aspect["aspect_term"],
                "sentiment": polarity,
                "start": int(aspect["start"]),
                "end": int(aspect["end"]),
            }
            for aspect, polarity in zip(aspects, polarities)
        ]
