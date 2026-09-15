"""Create exploratory-analysis tables and figures for the laptop dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    outputs_dir = PROJECT_ROOT / "outputs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sentences = pd.read_csv(processed_dir / "laptop_review_sentences.csv")
    aspects = pd.read_csv(processed_dir / "laptop_review_aspects.csv")

    aspects["aspect_normalized"] = (
        aspects["aspect_term"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    aspects["aspect_normalized"].value_counts().head(25).rename_axis("aspect_term").reset_index(
        name="count"
    ).to_csv(tables_dir / "top_aspects.csv", index=False)
    aspects["polarity"].value_counts().rename_axis("polarity").reset_index(name="count").to_csv(
        tables_dir / "polarity_distribution.csv", index=False
    )
    sentences["aspect_count"].value_counts().sort_index().rename_axis("aspect_count").reset_index(
        name="sentence_count"
    ).to_csv(tables_dir / "aspects_per_sentence.csv", index=False)

    summary = {
        "sentence_count": len(sentences),
        "sentences_with_aspects": int(sentences["has_aspect"].sum()),
        "sentences_without_aspects": int((~sentences["has_aspect"]).sum()),
        "aspect_annotation_count": len(aspects),
        "unique_aspect_terms": int(aspects["aspect_normalized"].nunique()),
        "average_sentence_length_characters": round(float(sentences["sentence_text"].str.len().mean()), 2),
        "average_aspects_per_annotated_sentence": round(
            float(aspects.groupby("sentence_id").size().mean()), 2
        ),
    }
    (outputs_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    order = aspects["polarity"].value_counts().index
    sns.countplot(data=aspects, x="polarity", order=order, color="#376996")
    plt.title("Aspect Sentiment Distribution")
    plt.xlabel("Polarity")
    plt.ylabel("Aspect annotations")
    plt.tight_layout()
    plt.savefig(figures_dir / "polarity_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=sentences, x="aspect_count", color="#7aa6c2")
    plt.title("Number of Annotated Aspects per Sentence")
    plt.xlabel("Aspect count")
    plt.ylabel("Sentences")
    plt.tight_layout()
    plt.savefig(figures_dir / "aspects_per_sentence.png", dpi=200)
    plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
