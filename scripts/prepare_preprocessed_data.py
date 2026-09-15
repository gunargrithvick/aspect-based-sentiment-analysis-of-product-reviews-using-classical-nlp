"""Create token-level and sequence-level preprocessed dataset files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.text_preprocessing import (
    align_aspects_to_bio,
    preprocess_text,
    token_records,
)


def main() -> None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    outputs_dir = PROJECT_ROOT / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sentences = pd.read_csv(processed_dir / "laptop_review_sentences.csv")
    aspects = pd.read_csv(processed_dir / "laptop_review_aspects.csv")
    aspects_by_sentence = {
        sentence_id: group.to_dict("records")
        for sentence_id, group in aspects.groupby("sentence_id", sort=False)
    }

    token_rows: list[dict] = []
    sequence_rows: list[dict] = []
    for sentence in sentences.itertuples(index=False):
        tokens = preprocess_text(sentence.sentence_text)
        sentence_aspects = aspects_by_sentence.get(sentence.sentence_id, [])
        labels = align_aspects_to_bio(tokens, sentence_aspects)
        records = token_records(tokens, labels)
        for record in records:
            token_rows.append({"sentence_id": sentence.sentence_id, **record})
        sequence_rows.append(
            {
                "sentence_id": sentence.sentence_id,
                "sentence_text": sentence.sentence_text,
                "tokens": records,
                "aspects": sentence_aspects,
            }
        )

    pd.DataFrame(token_rows).to_csv(processed_dir / "laptop_review_tokens.csv", index=False)
    with (processed_dir / "laptop_review_sequences.jsonl").open("w", encoding="utf-8") as file:
        for row in sequence_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    token_frame = pd.DataFrame(token_rows)
    summary = {
        "sentence_count": len(sentences),
        "token_count": len(token_frame),
        "aspect_token_count": int((token_frame["bio_label"] != "O").sum()),
        "negated_token_count": int(token_frame["negated"].sum()),
        "pos_distribution": {
            str(label): int(count)
            for label, count in token_frame["pos"].value_counts().sort_index().items()
        },
    }
    (outputs_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
