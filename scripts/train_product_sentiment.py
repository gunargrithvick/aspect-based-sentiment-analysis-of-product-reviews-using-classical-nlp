"""Train the Amazon product-review fallback and report held-out metrics."""

import argparse
import json
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classical_absa.product_sentiment import predict_product_sentiment, word_features

SOURCE = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
MEMBER = "sentiment labelled sentences/amazon_cells_labelled.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the UCI ZIP.")
    args = parser.parse_args()
    if args.archive:
        content = args.archive.read_bytes()
    else:
        with urlopen(SOURCE, timeout=30) as response:
            content = response.read()
    with ZipFile(BytesIO(content)) as archive:
        rows = archive.read(MEMBER).decode("utf-8").splitlines()
    labels_by_text = {}
    for row in rows:
        text, label = row.rsplit("\t", 1)
        normalized = " ".join(text.lower().split())
        labels_by_text.setdefault(normalized, set()).add(label)
    texts = [text for text, labels in labels_by_text.items() if len(labels) == 1]
    labels = ["positive" if labels_by_text[text] == {"1"} else "negative" for text in texts]
    train, test, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels,
    )
    vectorizer = DictVectorizer()
    train_features = vectorizer.fit_transform([word_features(text) for text in train])
    classifier = LogisticRegression(random_state=42).fit(train_features, train_labels)
    model = {
        "source": SOURCE,
        "source_sha256": sha256(content).hexdigest(),
        "intercept": float(classifier.intercept_[0]),
        "weights": dict(zip(vectorizer.get_feature_names_out(), map(float, classifier.coef_[0]))),
    }
    predictions = classifier.predict(vectorizer.transform([word_features(text) for text in test]))
    selected = [
        (gold, prediction) for text, gold in zip(test, test_labels)
        if (prediction := predict_product_sentiment(text, model)) is not None
    ]
    metrics = {
        "train_size": len(train),
        "test_size": len(test),
        "accuracy": accuracy_score(test_labels, predictions),
        "macro_f1": f1_score(test_labels, predictions, average="macro"),
        "fallback_coverage": len(selected) / len(test),
        "fallback_accuracy": sum(gold == prediction for gold, prediction in selected) / max(1, len(selected)),
    }
    model["evaluation"] = metrics
    (ROOT / "models" / "product_sentiment.json").write_text(json.dumps(model, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
