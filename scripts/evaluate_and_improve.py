"""Tune, ablate, and analyze the classical ABSA models."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.aspect_extractor import CRFAspectExtractor
from classical_absa.evaluation_metrics import (
    end_to_end_metrics,
    sentiment_metrics,
    span_metrics,
)
from classical_absa.sentiment_classifier import (
    TfidfSvmSentimentClassifier,
)
from classical_absa.text_preprocessing import (
    align_aspects_to_bio,
    preprocess_text,
)


def split_ids(sentences: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Return the fixed outer train/validation split."""

    train_ids, validation_ids = train_test_split(
        sentences["sentence_id"],
        test_size=0.2,
        random_state=42,
        stratify=sentences["has_aspect"],
    )
    return set(train_ids), set(validation_ids)


def build_sequences(
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
    for sentence in sentences[sentences["sentence_id"].isin(sentence_ids)].itertuples(
        index=False
    ):
        tokens = preprocess_text(sentence.sentence_text)
        labels = align_aspects_to_bio(tokens, aspects_by_sentence.get(sentence.sentence_id, []))
        token_sequences.append(tokens)
        label_sequences.append(labels)
    return token_sequences, label_sequences


def predict_aspects(
    sentences: pd.DataFrame, sentence_ids: set[str], extractor: CRFAspectExtractor
) -> pd.DataFrame:
    records: list[dict] = []
    for sentence in sentences[sentences["sentence_id"].isin(sentence_ids)].itertuples(
        index=False
    ):
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


def aspect_score(
    gold: pd.DataFrame, predicted: pd.DataFrame
) -> dict[str, float | int]:
    gold_spans = [
        (str(row.sentence_id), int(row.start), int(row.end))
        for row in gold.itertuples(index=False)
    ]
    predicted_spans = [
        (str(row.sentence_id), int(row.start), int(row.end))
        for row in predicted.itertuples(index=False)
    ]
    return span_metrics(gold_spans, predicted_spans)


def tune_crf(
    sentences: pd.DataFrame,
    aspects: pd.DataFrame,
    outer_train_ids: set[str],
) -> tuple[dict, list[dict]]:
    pool = sentences[sentences["sentence_id"].isin(outer_train_ids)].copy()
    inner_train, inner_valid = train_test_split(
        pool["sentence_id"],
        test_size=0.2,
        random_state=43,
        stratify=pool["has_aspect"],
    )
    inner_train_ids = set(inner_train)
    inner_valid_ids = set(inner_valid)
    inner_gold = aspects[aspects["sentence_id"].isin(inner_valid_ids)]
    configurations = [
        {"c1": 0.0, "c2": 0.1},
        {"c1": 0.1, "c2": 0.1},
        {"c1": 0.1, "c2": 1.0},
    ]
    scores: list[dict] = []
    for configuration in configurations:
        token_sequences, label_sequences = build_sequences(
            sentences, aspects, inner_train_ids
        )
        model = CRFAspectExtractor(**configuration).fit(token_sequences, label_sequences)
        predictions = predict_aspects(sentences, inner_valid_ids, model)
        metric = aspect_score(inner_gold, predictions)
        scores.append({**configuration, "validation_f1": metric["f1"]})
    best = max(scores, key=lambda row: row["validation_f1"])
    return {"c1": best["c1"], "c2": best["c2"]}, scores


def tune_svm(
    aspects: pd.DataFrame, outer_train_ids: set[str]
) -> tuple[float, list[dict]]:
    pool = aspects[aspects["sentence_id"].isin(outer_train_ids)].copy()
    train_ids, valid_ids = train_test_split(
        pool["sentence_id"].drop_duplicates(),
        test_size=0.2,
        random_state=43,
    )
    train_aspects = pool[pool["sentence_id"].isin(set(train_ids))]
    valid_aspects = pool[pool["sentence_id"].isin(set(valid_ids))]
    scores: list[dict] = []
    for c in (0.25, 1.0, 4.0):
        model = TfidfSvmSentimentClassifier(c=c).fit(train_aspects)
        predictions = model.predict(valid_aspects)
        metrics = sentiment_metrics(valid_aspects["polarity"], predictions)
        scores.append({"c": c, "validation_macro_f1": metrics["macro_f1"]})
    best = max(scores, key=lambda row: row["validation_macro_f1"])
    return float(best["c"]), scores


def add_sentiment_predictions(
    aspects: pd.DataFrame, classifier: TfidfSvmSentimentClassifier
) -> pd.DataFrame:
    result = aspects.copy()
    if not result.empty:
        result["polarity"] = classifier.predict(result)
    return result


def error_analysis(gold: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Categorize exact-span, boundary, missed, false-positive, and polarity errors."""

    gold_by_sentence = {
        sentence_id: group.to_dict("records")
        for sentence_id, group in gold.groupby("sentence_id", sort=False)
    }
    predicted_by_sentence = {
        sentence_id: group.to_dict("records")
        for sentence_id, group in predicted.groupby("sentence_id", sort=False)
    }
    rows: list[dict] = []
    sentence_ids = set(gold_by_sentence) | set(predicted_by_sentence)
    for sentence_id in sentence_ids:
        gold_rows = gold_by_sentence.get(sentence_id, [])
        predicted_rows = predicted_by_sentence.get(sentence_id, [])
        matched_predictions: set[int] = set()
        for gold_row in gold_rows:
            exact_matches = [
                (index, row)
                for index, row in enumerate(predicted_rows)
                if int(row["start"]) == int(gold_row["start"])
                and int(row["end"]) == int(gold_row["end"])
            ]
            if exact_matches:
                index, predicted_row = exact_matches[0]
                matched_predictions.add(index)
                if str(predicted_row.get("polarity", "")) != str(gold_row["polarity"]):
                    rows.append(
                        {
                            "sentence_id": sentence_id,
                            "sentence_text": gold_row["sentence_text"],
                            "error_type": "wrong_sentiment",
                            "gold_aspect": gold_row["aspect_term"],
                            "predicted_aspect": predicted_row["aspect_term"],
                            "gold_polarity": gold_row["polarity"],
                            "predicted_polarity": predicted_row.get("polarity", ""),
                        }
                    )
                continue

            overlaps = [
                (index, row)
                for index, row in enumerate(predicted_rows)
                if int(row["start"]) < int(gold_row["end"])
                and int(row["end"]) > int(gold_row["start"])
            ]
            if overlaps:
                index, predicted_row = overlaps[0]
                matched_predictions.add(index)
                error_type = "boundary_error"
                predicted_term = predicted_row["aspect_term"]
            else:
                error_type = "missed_aspect"
                predicted_term = ""
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "sentence_text": gold_row["sentence_text"],
                    "error_type": error_type,
                    "gold_aspect": gold_row["aspect_term"],
                    "predicted_aspect": predicted_term,
                    "gold_polarity": gold_row["polarity"],
                    "predicted_polarity": "",
                }
            )

        for index, predicted_row in enumerate(predicted_rows):
            if index not in matched_predictions:
                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "sentence_text": predicted_row["sentence_text"],
                        "error_type": "false_positive_aspect",
                        "gold_aspect": "",
                        "predicted_aspect": predicted_row["aspect_term"],
                        "gold_polarity": "",
                        "predicted_polarity": predicted_row.get("polarity", ""),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "sentence_id",
            "sentence_text",
            "error_type",
            "gold_aspect",
            "predicted_aspect",
            "gold_polarity",
            "predicted_polarity",
        ],
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
    outer_train_ids, outer_valid_ids = split_ids(sentences)
    train_aspects = aspects[aspects["sentence_id"].isin(outer_train_ids)].copy()
    valid_aspects = aspects[aspects["sentence_id"].isin(outer_valid_ids)].copy()

    best_crf, crf_tuning = tune_crf(sentences, aspects, outer_train_ids)
    best_svm, svm_tuning = tune_svm(aspects, outer_train_ids)
    token_sequences, label_sequences = build_sequences(sentences, aspects, outer_train_ids)
    tuned_crf = CRFAspectExtractor(**best_crf).fit(token_sequences, label_sequences)
    tuned_svm = TfidfSvmSentimentClassifier(c=best_svm).fit(train_aspects)

    tuned_predicted = predict_aspects(sentences, outer_valid_ids, tuned_crf)
    tuned_predicted = add_sentiment_predictions(tuned_predicted, tuned_svm)
    tuned_aspect_metrics = aspect_score(valid_aspects, tuned_predicted)
    tuned_sentiment_predictions = tuned_svm.predict(valid_aspects)
    tuned_sentiment_metrics = sentiment_metrics(valid_aspects["polarity"], tuned_sentiment_predictions)

    gold_end_to_end = [
        (str(row.sentence_id), int(row.start), int(row.end), str(row.polarity))
        for row in valid_aspects.itertuples(index=False)
    ]
    predicted_end_to_end = [
        (str(row.sentence_id), int(row.start), int(row.end), str(row.polarity))
        for row in tuned_predicted.itertuples(index=False)
    ]
    tuned_result = {
        "tuning": {"crf": crf_tuning, "svm": svm_tuning},
        "selected_parameters": {"crf": best_crf, "svm_c": best_svm},
        "aspect_extraction": tuned_aspect_metrics,
        "sentiment_classification_on_gold_aspects": tuned_sentiment_metrics,
        "end_to_end": end_to_end_metrics(gold_end_to_end, predicted_end_to_end),
    }

    lexical_crf = CRFAspectExtractor(
        c1=best_crf["c1"], c2=best_crf["c2"], feature_profile="lexical_only"
    ).fit(token_sequences, label_sequences)
    lexical_predicted = predict_aspects(sentences, outer_valid_ids, lexical_crf)
    ablation = {
        "full_features": tuned_aspect_metrics,
        "lexical_only_features": aspect_score(valid_aspects, lexical_predicted),
    }

    errors = error_analysis(valid_aspects, tuned_predicted)
    errors.to_csv(outputs_dir / "error_analysis.csv", index=False)
    error_counts = Counter(errors["error_type"]) if not errors.empty else Counter()
    examples = {}
    for error_type in error_counts:
        examples[error_type] = errors[errors["error_type"] == error_type].head(5).to_dict(
            "records"
        )
    (outputs_dir / "error_analysis_summary.json").write_text(
        json.dumps(
            {
                "error_counts": dict(error_counts),
                "examples": examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (outputs_dir / "tuned_results.json").write_text(
        json.dumps(tuned_result, indent=2), encoding="utf-8"
    )
    (outputs_dir / "feature_ablation.json").write_text(
        json.dumps(ablation, indent=2), encoding="utf-8"
    )
    tuned_predicted.to_csv(outputs_dir / "tuned_validation_predictions.csv", index=False)
    joblib.dump(tuned_crf, models_dir / "tuned_crf_aspect_extractor.joblib")
    joblib.dump(tuned_svm, models_dir / "tuned_svm_sentiment_classifier.joblib")

    baseline = json.loads((outputs_dir / "baseline_results.json").read_text(encoding="utf-8"))
    original_improved = json.loads(
        (outputs_dir / "improved_results.json").read_text(encoding="utf-8")
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "baseline_lexicon_logistic_regression",
                "aspect_f1": baseline["aspect_extraction"]["f1"],
                "sentiment_macro_f1": baseline["sentiment_classification_on_gold_aspects"]["macro_f1"],
                "end_to_end_f1": baseline["end_to_end"]["f1"],
            },
            {
                "model": "crf_svm_default",
                "aspect_f1": original_improved["aspect_extraction"]["f1"],
                "sentiment_macro_f1": original_improved["sentiment_classification_on_gold_aspects"]["macro_f1"],
                "end_to_end_f1": original_improved["end_to_end"]["f1"],
            },
            {
                "model": "crf_svm_tuned",
                "aspect_f1": tuned_result["aspect_extraction"]["f1"],
                "sentiment_macro_f1": tuned_result["sentiment_classification_on_gold_aspects"]["macro_f1"],
                "end_to_end_f1": tuned_result["end_to_end"]["f1"],
            },
            {
                "model": "crf_lexical_only_ablation",
                "aspect_f1": ablation["lexical_only_features"]["f1"],
                "sentiment_macro_f1": "not_evaluated",
                "end_to_end_f1": "not_evaluated",
            },
        ]
    )
    comparison.to_csv(reports_dir / "final_model_comparison.csv", index=False)
    print(json.dumps(tuned_result, indent=2))
    print(json.dumps({"ablation": ablation, "error_counts": dict(error_counts)}, indent=2))


if __name__ == "__main__":
    main()
