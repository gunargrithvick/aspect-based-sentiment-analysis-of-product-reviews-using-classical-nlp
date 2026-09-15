"""Metrics for aspect extraction, polarity classification, and end-to-end output."""

from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def span_metrics(gold_spans: Iterable[tuple[str, int, int]], predicted_spans: Iterable[tuple[str, int, int]]) -> dict[str, float | int]:
    """Calculate exact character-span precision, recall, and F1."""

    gold = set(gold_spans)
    predicted = set(predicted_spans)
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "true_positive_count": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sentiment_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict:
    """Return standard multi-class sentiment metrics and a confusion matrix."""

    true_values = list(y_true)
    predicted_values = list(y_pred)
    labels = sorted(set(true_values) | set(predicted_values))
    return {
        "accuracy": accuracy_score(true_values, predicted_values),
        "macro_f1": f1_score(true_values, predicted_values, average="macro", zero_division=0),
        "labels": labels,
        "classification_report": classification_report(
            true_values, predicted_values, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(true_values, predicted_values, labels=labels).tolist(),
    }


def end_to_end_metrics(
    gold_records: Iterable[tuple[str, int, int, str]],
    predicted_records: Iterable[tuple[str, int, int, str]],
) -> dict[str, float | int]:
    """Evaluate exact `(sentence, aspect span, polarity)` predictions."""

    return span_metrics(gold_records, predicted_records)
