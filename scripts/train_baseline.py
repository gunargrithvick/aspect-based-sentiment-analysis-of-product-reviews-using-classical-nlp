"""Train and evaluate the classical baseline system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.baseline_models import (
    AspectLexiconExtractor,
    TfidfLogisticSentimentClassifier,
)
from classical_absa.evaluation_metrics import (
    end_to_end_metrics,
    sentiment_metrics,
    span_metrics,
)


def _records_with_predictions(
    sentence_ids: set[str],
    sentences: pd.DataFrame,
    extractor: AspectLexiconExtractor,
    sentiment_classifier: TfidfLogisticSentimentClassifier,
) -> list[dict]:
    sentence_lookup = sentences.set_index("sentence_id")["sentence_text"].to_dict()
    records: list[dict] = []
    for sentence_id in sentence_ids:
        text = sentence_lookup[sentence_id]
        for aspect in extractor.predict(text):
            records.append({"sentence_id": sentence_id, "sentence_text": text, **aspect})

    if not records:
        return records
    frame = pd.DataFrame(records)
    frame["polarity"] = sentiment_classifier.predict(frame)
    return frame.to_dict("records")


def main() -> None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sentences = pd.read_csv(processed_dir / "laptop_review_sentences.csv")
    aspects = pd.read_csv(processed_dir / "laptop_review_aspects.csv")
    sentences["sentence_id"] = sentences["sentence_id"].astype(str)
    aspects["sentence_id"] = aspects["sentence_id"].astype(str)
    sentence_ids = sentences["sentence_id"]
    train_ids, validation_ids = train_test_split(
        sentence_ids,
        test_size=0.2,
        random_state=42,
        stratify=sentences["has_aspect"],
    )
    train_ids = set(train_ids)
    validation_ids = set(validation_ids)

    train_aspects = aspects[aspects["sentence_id"].isin(train_ids)].copy()
    validation_aspects = aspects[aspects["sentence_id"].isin(validation_ids)].copy()
    validation_sentences = sentences[sentences["sentence_id"].isin(validation_ids)]

    extractor = AspectLexiconExtractor(minimum_frequency=1).fit(train_aspects)
    sentiment_classifier = TfidfLogisticSentimentClassifier().fit(train_aspects)
    predictions = _records_with_predictions(
        validation_ids, validation_sentences, extractor, sentiment_classifier
    )
    predicted_frame = pd.DataFrame(predictions)

    gold_spans = [
        (str(row.sentence_id), int(row.start), int(row.end))
        for row in validation_aspects.itertuples(index=False)
    ]
    predicted_spans = [
        (str(row.sentence_id), int(row.start), int(row.end))
        for row in predicted_frame.itertuples(index=False)
    ] if not predicted_frame.empty else []

    gold_for_sentiment = validation_aspects.copy()
    gold_sentiments = sentiment_classifier.predict(gold_for_sentiment)
    sentiment_result = sentiment_metrics(gold_for_sentiment["polarity"], gold_sentiments)

    gold_end_to_end = [
        (str(row.sentence_id), int(row.start), int(row.end), str(row.polarity))
        for row in validation_aspects.itertuples(index=False)
    ]
    predicted_end_to_end = [
        (str(row.sentence_id), int(row.start), int(row.end), str(row.polarity))
        for row in predicted_frame.itertuples(index=False)
    ] if not predicted_frame.empty else []

    result = {
        "split": {
            "random_state": 42,
            "training_sentence_count": len(train_ids),
            "validation_sentence_count": len(validation_ids),
            "training_aspect_count": len(train_aspects),
            "validation_aspect_count": len(validation_aspects),
        },
        "aspect_extraction": span_metrics(gold_spans, predicted_spans),
        "sentiment_classification_on_gold_aspects": sentiment_result,
        "end_to_end": end_to_end_metrics(gold_end_to_end, predicted_end_to_end),
        "model": {
            "aspect_extractor": "training-split exact aspect lexicon",
            "sentiment_classifier": "TF-IDF word unigrams/bigrams with balanced Logistic Regression",
        },
    }
    (outputs_dir / "baseline_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    predicted_frame.to_csv(outputs_dir / "baseline_validation_predictions.csv", index=False)
    joblib.dump(extractor, models_dir / "baseline_aspect_lexicon.joblib")
    joblib.dump(sentiment_classifier, models_dir / "baseline_sentiment_classifier.joblib")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
