"""Aspect-level sentiment classification using traditional classifiers."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .feature_engineering import aspect_context


class TfidfSvmSentimentClassifier:
    """Classify aspect sentiment with TF-IDF features and a linear SVM."""

    def __init__(self, c: float = 1.0) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.classifier = LinearSVC(C=c, class_weight="balanced", random_state=42)

    @staticmethod
    def _contexts(aspect_frame: pd.DataFrame) -> list[str]:
        return [
            aspect_context(row.sentence_text, int(row.start), int(row.end))
            for row in aspect_frame.itertuples(index=False)
        ]

    def fit(self, aspect_frame: pd.DataFrame) -> TfidfSvmSentimentClassifier:
        contexts = self._contexts(aspect_frame)
        features = self.vectorizer.fit_transform(contexts)
        self.classifier.fit(features, aspect_frame["polarity"].astype(str))
        return self

    def predict(self, aspect_frame: pd.DataFrame) -> list[str]:
        features = self.vectorizer.transform(self._contexts(aspect_frame))
        return self.classifier.predict(features).tolist()
