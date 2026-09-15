import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_absa.inference_pipeline import AspectBasedSentimentAnalyzer


class FakeAspectExtractor:
    def predict(self, text):
        return [{"aspect_term": "screen", "start": 4, "end": 10}]


class FakeSentimentClassifier:
    def predict(self, frame: pd.DataFrame):
        assert list(frame["aspect_term"]) == ["screen"]
        return ["positive"]


def test_inference_pipeline_returns_structured_results():
    analyzer = AspectBasedSentimentAnalyzer(
        FakeAspectExtractor(), FakeSentimentClassifier()
    )

    result = analyzer.analyze("The screen is bright.")

    assert result == [
        {"aspect": "screen", "sentiment": "positive", "start": 4, "end": 10}
    ]
