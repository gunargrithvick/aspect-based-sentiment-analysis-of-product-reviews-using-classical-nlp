"""Standalone inference pipeline for aspect-based sentiment analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .product_sentiment import predict_product_sentiment
from .review_features import (
    ASPECT_DOMAINS,
    ASPECT_PATTERN,
    IMPLICIT_CUES,
    OPINIONS,
    PROTECTED_WORDS,
    SARCASM_PATTERN,
    clause_ranges,
    lexical_sentiment,
    normalize_spelling,
)


class AspectBasedSentimentAnalyzer:
    """Combine the trained CRF extractor and SVM sentiment classifier."""

    def __init__(self, aspect_extractor, sentiment_classifier, product_model: dict | None = None) -> None:
        self.aspect_extractor = aspect_extractor
        self.sentiment_classifier = sentiment_classifier
        self.product_model = product_model
        vocabulary = (product_model or {}).get("weights", {})
        self.known_words = frozenset(vocabulary) | PROTECTED_WORDS | ENGLISH_STOP_WORDS | OPINIONS.keys()
        self.known_words |= {word for term in ASPECT_DOMAINS for word in term.split()}

    @classmethod
    def from_model_directory(cls, model_directory: str | Path) -> AspectBasedSentimentAnalyzer:
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
        product_path = directory / "product_sentiment.json"
        product_model = json.loads(product_path.read_text(encoding="utf-8")) if product_path.is_file() else None
        return cls(joblib.load(extractor_path), joblib.load(classifier_path), product_model)

    def analyze(self, text: str) -> list[dict]:
        """Return detected aspects, with original offsets for explicit mentions."""
        return self.analyze_details(text)["results"]

    def analyze_details(self, text: str) -> dict:
        """Analyze supported categories and report uncertain or inferred results."""
        if not text or not text.strip():
            raise ValueError("Review text must not be empty.")

        normalized, corrections, boundaries = normalize_spelling(text, self.known_words)
        lexical_aspects = {
            match.span(): ASPECT_DOMAINS[match.group().lower()]
            for match in ASPECT_PATTERN.finditer(normalized)
        }
        clauses = clause_ranges(normalized)
        inferred = [
            (index, aspect, domain, sentiment)
            for index, (start, end) in enumerate(clauses)
            for aspect, domain, sentiment, pattern in IMPLICIT_CUES
            if re.search(pattern, normalized[start:end], re.IGNORECASE)
        ]
        domains = set(lexical_aspects.values()) | {domain for _, _, domain, _ in inferred}
        specific_domains = domains - {"electronics"}
        use_crf = not specific_domains or "computers" in specific_domains
        spans = dict.fromkeys(lexical_aspects)
        if domains and use_crf:
            for candidate in self.aspect_extractor.predict(normalized):
                start, end = int(candidate["start"]), int(candidate["end"])
                if any(start < right and end > left for left, right in lexical_aspects):
                    continue
                spans[start, end] = None

        ordered_spans = sorted(spans)
        model_predictions = {}
        if ordered_spans and use_crf:
            frame = pd.DataFrame([
                {"sentence_text": normalized, "aspect_term": normalized[start:end], "start": start, "end": end}
                for start, end in ordered_spans
            ])
            model_predictions = dict(zip(ordered_spans, self.sentiment_classifier.predict(frame)))

        results = []
        sarcasm_detected = False
        for index, (left, right) in enumerate(clauses):
            targets = [(start, end) for start, end in ordered_spans if left <= start < end <= right]
            centers = [(start + end) / 2 - left for start, end in targets]
            clause = normalized[left:right]
            possible_sarcasm = bool(SARCASM_PATTERN.search(clause))
            sarcasm_detected |= possible_sarcasm
            for (start, end), center in zip(targets, centers):
                polarity = lexical_sentiment(clause, center, centers)
                if polarity is None:
                    polarity = model_predictions.get((start, end))
                if polarity is None and self.product_model and len(targets) == 1 and not possible_sarcasm:
                    polarity = predict_product_sentiment(clause, self.product_model)
                original_start, original_end = boundaries[start], boundaries[end]
                results.append({
                    "aspect": text[original_start:original_end],
                    "sentiment": polarity if polarity and not possible_sarcasm else "uncertain",
                    "start": original_start,
                    "end": original_end,
                })
            if not targets:
                for clause_index, aspect, _, sentiment in inferred:
                    if clause_index == index:
                        results.append({
                            "aspect": aspect, "sentiment": "uncertain" if possible_sarcasm else sentiment,
                            "start": None, "end": None, "implicit": True,
                        })

        return {
            "results": results,
            "corrections": corrections,
            "domains": sorted(specific_domains or domains),
            "possible_sarcasm": sarcasm_detected,
        }
