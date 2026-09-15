"""Train and evaluate the CRF plus SVM classical NLP system."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.aspect_extractor import CRFAspectExtractor  # noqa: E402
from classical_absa.evaluation_metrics import (  # noqa: E402
    end_to_end_metrics,
    sentiment_metrics,
    span_metrics,
)
from classical_absa.sentiment_classifier import TfidfSvmSentimentClassifier  # noqa: E402
from classical_absa.text_preprocessing import (  # noqa: E402
    align_aspects_to_bio,
    preprocess_text,
)


def split_sentence_ids(sentences: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Create the fixed sentence-level split used by the baseline."""

    sentence_ids = sentences["sentence_id"].astype(str)
    train_ids, validation_ids = train_test_split(
        sentence_ids,
        test_size=0.2,
        random_state=42,
        stratify=sentences["has_aspect"],
    )
    return set(train_ids), set(validation_ids)


def build_training_sequences(
    sentences: pd.DataFrame, aspects: pd.DataFrame, sentence_ids: set[str]
) -> tuple[list, list]:
    aspects_by_sentence = {
        sentence_id: group.to_dict("records")
        for sentence_id, group in aspects[aspects["sentence_id"].isin(sentence_ids)].groupby(
            "sentence_id", sort=False
        )
    }
    token_sequences = []
    label_sequences = []
    training_sentences = sentences[sentences["sentence_id"].isin(sentence_ids)]
    for sentence in training_sentences.itertuples(index=False):
        tokens = preprocess_text(sentence.sentence_text)
        labels = align_aspects_to_bio(tokens, aspects_by_sentence.get(sentence.sentence_id, []))
        token_sequences.append(tokens)
        label_sequences.append(labels)
    return token_sequences, label_sequences


def predict_aspects(
    sentence_ids: set[str], sentences: pd.DataFrame, extractor: CRFAspectExtractor
) -> pd.DataFrame:
    records: list[dict] = []
    for sentence in sentences[sentences["sentence_id"].isin(sentence_ids)].itertuples(index=False):
        for aspect in extractor.predict(sentence.sentence_text):
            records.append(
                {
                    "sentence_id": sentence.sentence_id,
                    "sentence_text": sentence.sentence_text,
                    **aspect,
                }
            )
    return pd.DataFrame(
        records,
        columns=["sentence_id", "sentence_text", "aspect_term", "start", "end"],
    )


def main() -> None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs"
    reports_dir = PROJECT_ROOT / "reports" / "tables"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sentences = pd.read_csv(processed_dir / "laptop_review_sentences.csv")
    aspects = pd.read_csv(processed_dir / "laptop_review_aspects.csv")
    sentences["sentence_id"] = sentences["sentence_id"].astype(str)
    aspects["sentence_id"] = aspects["sentence_id"].astype(str)
    train_ids, validation_ids = split_sentence_ids(sentences)
    train_aspects = aspects[aspects["sentence_id"].isin(train_ids)].copy()
    validation_aspects = aspects[aspects["sentence_id"].isin(validation_ids)].copy()

    token_sequences, label_sequences = build_training_sequences(sentences, aspects, train_ids)
    aspect_extractor = CRFAspectExtractor().fit(token_sequences, label_sequences)
    sentiment_classifier = TfidfSvmSentimentClassifier().fit(train_aspects)

    predicted_aspects = predict_aspects(validation_ids, sentences, aspect_extractor)
    if not predicted_aspects.empty:
        predicted_aspects["polarity"] = sentiment_classifier.predict(predicted_aspects)
    else:
        predicted_aspects["polarity"] = pd.Series(dtype=str)

    gold_spans = [
        (str(row.sentence_id), int(row.start), int(row.end))
        for row in validation_aspects.itertuples(index=False)
    ]
    predicted_spans = [
        (str(row.sentence_id), int(row.start), int(row.end))
        for row in predicted_aspects.itertuples(index=False)
    ]

    gold_sentiments = sentiment_classifier.predict(validation_aspects)
    sentiment_result = sentiment_metrics(validation_aspects["polarity"], gold_sentiments)
    gold_end_to_end = [
        (str(row.sentence_id), int(row.start), int(row.end), str(row.polarity))
        for row in validation_aspects.itertuples(index=False)
    ]
    predicted_end_to_end = [
        (str(row.sentence_id), int(row.start), int(row.end), str(row.polarity))
        for row in predicted_aspects.itertuples(index=False)
    ]

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
            "aspect_extractor": "CRF with lexical, POS, context, shape, and negation features",
            "sentiment_classifier": "TF-IDF word unigrams/bigrams with balanced Linear SVM",
        },
    }
    (outputs_dir / "improved_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    predicted_aspects.to_csv(outputs_dir / "improved_validation_predictions.csv", index=False)
    joblib.dump(aspect_extractor, models_dir / "crf_aspect_extractor.joblib")
    joblib.dump(sentiment_classifier, models_dir / "svm_sentiment_classifier.joblib")

    comparison = [
        {
            "model": "baseline_lexicon_logistic_regression",
            "aspect_extraction_f1": json.loads(
                (outputs_dir / "baseline_results.json").read_text(encoding="utf-8")
            )["aspect_extraction"]["f1"],
            "sentiment_macro_f1": json.loads(
                (outputs_dir / "baseline_results.json").read_text(encoding="utf-8")
            )["sentiment_classification_on_gold_aspects"]["macro_f1"],
            "end_to_end_f1": json.loads(
                (outputs_dir / "baseline_results.json").read_text(encoding="utf-8")
            )["end_to_end"]["f1"],
        },
        {
            "model": "crf_svm",
            "aspect_extraction_f1": result["aspect_extraction"]["f1"],
            "sentiment_macro_f1": result["sentiment_classification_on_gold_aspects"]["macro_f1"],
            "end_to_end_f1": result["end_to_end"]["f1"],
        },
    ]
    pd.DataFrame(comparison).to_csv(reports_dir / "model_comparison.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
