"""Deterministic classical preprocessing for review sentences.

The first preprocessing layer intentionally uses standard-library rules so it
can run without downloading a pretrained language model or an external NLP
resource. Character offsets are preserved because the SemEval annotations are
defined over the original sentence text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)?|[^\w\s]",
    flags=re.UNICODE,
)

NEGATION_WORDS = {
    "cannot",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hardly",
    "neither",
    "never",
    "no",
    "nor",
    "not",
    "scarcely",
    "shouldn't",
    "wasn't",
    "weren't",
    "without",
    "won't",
    "wouldn't",
}
NEGATION_SCOPE_BREAKS = {".", ",", ";", ":", "!", "?", "but", "however", "although", "though"}


@dataclass(frozen=True)
class Token:
    """A token and its deterministic linguistic features."""

    text: str
    start: int
    end: int
    normalized: str
    lemma: str
    pos: str
    negated: bool


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text while preserving each token's character span."""

    return [(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def normalize_token(token: str) -> str:
    """Normalize case and curly apostrophes without removing useful content."""

    return token.lower().replace("’", "'")


def heuristic_pos(token: str) -> str:
    """Assign a lightweight POS tag using deterministic lexical rules.

    These tags are deliberately coarse and are intended as baseline features.
    A statistical POS tagger can be evaluated later as a separate experiment.
    """

    normalized = normalize_token(token)
    if not normalized:
        return "X"
    if all(not character.isalnum() for character in normalized):
        return "PUNCT"
    if normalized.replace(".", "", 1).isdigit():
        return "NUM"
    if normalized in {"a", "an", "the"}:
        return "DET"
    if normalized in {"i", "you", "he", "she", "it", "we", "they", "this", "that"}:
        return "PRON"
    if normalized in {"and", "or", "but", "because", "although", "though", "however"}:
        return "CONJ"
    if normalized in {"in", "on", "at", "for", "from", "with", "of", "to", "by", "without"}:
        return "ADP"
    if normalized.endswith("ly"):
        return "ADV"
    if normalized.endswith(("ing", "ed", "en")):
        return "VERB"
    if normalized.endswith(("ous", "ful", "able", "ible", "ive", "al", "ic", "ish")):
        return "ADJ"
    if normalized.endswith(("s", "'s")) and len(normalized) > 3:
        return "NOUN"
    return "X"


def rule_based_lemma(token: str, pos: str) -> str:
    """Apply conservative suffix rules for a baseline lemma."""

    word = normalize_token(token)
    if len(word) <= 3 or pos in {"PUNCT", "NUM", "X"}:
        return word
    if pos == "VERB":
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("ing") and len(word) > 5:
            return word[:-3]
        if word.endswith("ed") and len(word) > 4:
            return word[:-2]
        if word.endswith("en") and len(word) > 4:
            return word[:-2]
    if pos == "NOUN" and word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if pos == "NOUN" and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def add_negation_features(tokens: Sequence[tuple[str, int, int]]) -> list[bool]:
    """Mark tokens occurring inside a simple negation scope."""

    negated: list[bool] = []
    active = False
    for token, _, _ in tokens:
        normalized = normalize_token(token)
        if normalized in NEGATION_WORDS:
            active = True
            negated.append(False)
            continue
        if normalized in NEGATION_SCOPE_BREAKS:
            active = False
            negated.append(False)
            continue
        negated.append(active)
    return negated


def preprocess_text(text: str) -> list[Token]:
    """Return token-level normalized, lemma, POS, and negation features."""

    raw_tokens = tokenize_with_offsets(text)
    negated = add_negation_features(raw_tokens)
    processed: list[Token] = []
    for (token, start, end), is_negated in zip(raw_tokens, negated):
        normalized = normalize_token(token)
        pos = heuristic_pos(token)
        processed.append(
            Token(
                text=token,
                start=start,
                end=end,
                normalized=normalized,
                lemma=rule_based_lemma(token, pos),
                pos=pos,
                negated=is_negated,
            )
        )
    return processed


def align_aspects_to_bio(
    tokens: Sequence[Token], aspects: Iterable[dict]
) -> list[str]:
    """Align character-offset aspects to token-level BIO labels."""

    labels = ["O"] * len(tokens)
    sorted_aspects = sorted(aspects, key=lambda item: (int(item["start"]), int(item["end"])))
    for aspect in sorted_aspects:
        start = int(aspect["start"])
        end = int(aspect["end"])
        matching_indices = [
            index
            for index, token in enumerate(tokens)
            if token.start < end and token.end > start
        ]
        for position, index in enumerate(matching_indices):
            labels[index] = "B-ASPECT" if position == 0 else "I-ASPECT"
    return labels


def token_records(tokens: Sequence[Token], bio_labels: Sequence[str]) -> list[dict]:
    """Convert token objects into serializable records."""

    if len(tokens) != len(bio_labels):
        raise ValueError("The number of BIO labels must match the number of tokens.")
    return [{**asdict(token), "bio_label": label} for token, label in zip(tokens, bio_labels)]
