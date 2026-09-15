"""Parse and validate the SemEval laptop-review XML dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.dataset_loader import load_semeval_xml, validate_annotations  # noqa: E402


def main() -> None:
    raw_file = PROJECT_ROOT / "data" / "raw" / "Laptop_Train_v2.xml"
    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    outputs_dir = PROJECT_ROOT / "outputs"

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sentences, aspects = load_semeval_xml(raw_file)
    validation = validate_annotations(sentences, aspects)
    if not validation["is_valid"]:
        raise ValueError(f"Dataset validation failed: {validation}")

    sentences.to_csv(processed_dir / "laptop_review_sentences.csv", index=False)
    aspects.to_csv(processed_dir / "laptop_review_aspects.csv", index=False)

    summary = {
        "source_file": str(raw_file.relative_to(PROJECT_ROOT)),
        "validation": validation,
        "polarity_distribution": {
            str(label): int(count)
            for label, count in aspects["polarity"].value_counts().sort_index().items()
        },
    }
    (outputs_dir / "dataset_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
