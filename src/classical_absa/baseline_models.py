"""Transparent rule-based and statistical baseline models."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .feature_engineering import aspect_context
from .text_preprocessing import normalize_token


class AspectLexiconExtractor:
    """Extract exact aspect terms seen in the training split.

    This is a deliberately simple gazetteer baseline. The lexicon must be fit
    only on the training partition to avoid leaking validation annotations.
    """

    def __init__(self, minimum_frequency: int = 1) -> None:
        self.minimum_frequency = minimum_frequency
        self.aspect_counts: Counter[str] = Counter()
        self.terms: list[str] = []

    def fit(self, aspect_frame: pd.DataFrame) -> AspectLexiconExtractor:
        counts = Counter(normalize_token(str(term)) for term in aspect_frame["aspect_term"])
        self.aspect_counts = counts
        self.terms = sorted(
            [term for term, count in counts.items() if count >= self.minimum_frequency],
            key=lambda term: (-len(term), -counts[term], term),
        )
        return self

    def predict(self, text: str) -> list[dict]:
        """Return non-overlapping exact lexicon matches with character offsets."""

        candidates: list[dict] = []
        for term in self.terms:
            if not term:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                candidates.append(
                    {
                        "aspect_term": text[match.start() : match.end()],
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        selected: list[dict] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (item["start"], -(item["end"] - item["start"])),
        ):
            overlaps = any(
                candidate["start"] < existing["end"]
                and candidate["end"] > existing["start"]
                for existing in selected
            )
            if not overlaps:
                selected.append(candidate)
        return sorted(selected, key=lambda item: item["start"])


class TfidfLogisticSentimentClassifier:
    """Classify aspect sentiment using TF-IDF and logistic regression."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    def fit(self, aspect_frame: pd.DataFrame) -> TfidfLogisticSentimentClassifier:
        contexts = [
            aspect_context(row.sentence_text, int(row.start), int(row.end))
            for row in aspect_frame.itertuples(index=False)
        ]
        features = self.vectorizer.fit_transform(contexts)
        self.classifier.fit(features, aspect_frame["polarity"].astype(str))
        return self

    def predict(self, aspect_frame: pd.DataFrame) -> list[str]:
        contexts = [
            aspect_context(row.sentence_text, int(row.start), int(row.end))
            for row in aspect_frame.itertuples(index=False)
        ]
        features = self.vectorizer.transform(contexts)
        return self.classifier.predict(features).tolist()
