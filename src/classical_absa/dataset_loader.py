"""Load and validate SemEval aspect-based sentiment annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import xml.etree.ElementTree as ET

import pandas as pd


def load_semeval_xml(xml_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a SemEval ABSA XML file.

    Returns a sentence-level dataframe and an aspect-level dataframe. Character
    offsets use the SemEval convention: ``from`` is inclusive and ``to`` is
    exclusive.
    """

    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    root = ET.parse(path).getroot()
    sentence_records: list[dict] = []
    aspect_records: list[dict] = []

    for sentence in root.findall("sentence"):
        sentence_id = sentence.get("id", "")
        text_element = sentence.find("text")
        text = text_element.text if text_element is not None and text_element.text else ""
        aspects_element = sentence.find("aspectTerms")
        aspect_elements = (
            [] if aspects_element is None else aspects_element.findall("aspectTerm")
        )

        sentence_records.append(
            {
                "sentence_id": sentence_id,
                "sentence_text": text,
                "has_aspect": bool(aspect_elements),
                "aspect_count": len(aspect_elements),
            }
        )

        for aspect in aspect_elements:
            start = int(aspect.get("from", "-1"))
            end = int(aspect.get("to", "-1"))
            term = aspect.get("term", "")
            extracted = text[start:end] if 0 <= start <= end <= len(text) else ""
            aspect_records.append(
                {
                    "sentence_id": sentence_id,
                    "sentence_text": text,
                    "aspect_term": term,
                    "polarity": aspect.get("polarity", ""),
                    "start": start,
                    "end": end,
                    "offset_text": extracted,
                    "offset_matches": extracted == term,
                }
            )

    sentences = pd.DataFrame(
        sentence_records,
        columns=["sentence_id", "sentence_text", "has_aspect", "aspect_count"],
    )
    aspects = pd.DataFrame(
        aspect_records,
        columns=[
            "sentence_id",
            "sentence_text",
            "aspect_term",
            "polarity",
            "start",
            "end",
            "offset_text",
            "offset_matches",
        ],
    )
    return sentences, aspects


def validate_annotations(
    sentences: pd.DataFrame, aspects: pd.DataFrame
) -> dict[str, int | bool]:
    """Return validation counts for parsed annotations."""

    duplicate_sentence_ids = int(sentences["sentence_id"].duplicated().sum())
    invalid_offsets = int((~aspects["offset_matches"]).sum()) if not aspects.empty else 0
    missing_text = int(sentences["sentence_text"].eq("").sum())
    unknown_sentence_ids = int(
        (~aspects["sentence_id"].isin(sentences["sentence_id"])).sum()
        if not aspects.empty
        else 0
    )
    return {
        "sentence_count": int(len(sentences)),
        "aspect_annotation_count": int(len(aspects)),
        "sentences_with_aspects": int(sentences["has_aspect"].sum()),
        "duplicate_sentence_ids": duplicate_sentence_ids,
        "invalid_offsets": invalid_offsets,
        "missing_text": missing_text,
        "unknown_sentence_ids": unknown_sentence_ids,
        "is_valid": all(
            value == 0
            for key, value in {
                "duplicate_sentence_ids": duplicate_sentence_ids,
                "invalid_offsets": invalid_offsets,
                "missing_text": missing_text,
                "unknown_sentence_ids": unknown_sentence_ids,
            }.items()
        ),
    }
