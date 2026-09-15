from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_absa.baseline_models import AspectLexiconExtractor  # noqa: E402
from classical_absa.feature_engineering import aspect_context  # noqa: E402


def test_aspect_lexicon_prefers_longest_non_overlapping_match():
    aspects = pd.DataFrame(
        [
            {"aspect_term": "battery", "polarity": "positive"},
            {"aspect_term": "battery life", "polarity": "positive"},
        ]
    )
    extractor = AspectLexiconExtractor().fit(aspects)

    predictions = extractor.predict("The battery life is excellent.")

    assert len(predictions) == 1
    assert predictions[0]["aspect_term"] == "battery life"


def test_aspect_context_marks_target_and_negation():
    context = aspect_context("The battery is not good.", start=4, end=11)

    assert "__ASPECT__" in context
    assert "NEG_good" in context
