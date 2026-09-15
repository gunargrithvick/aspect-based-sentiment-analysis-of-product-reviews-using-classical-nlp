from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_absa.text_preprocessing import (  # noqa: E402
    align_aspects_to_bio,
    preprocess_text,
    tokenize_with_offsets,
)


def test_token_offsets_reconstruct_original_non_whitespace_text():
    text = "The battery-life is excellent!"
    tokens = tokenize_with_offsets(text)
    reconstructed = "".join(text[start:end] for _, start, end in tokens)
    expected = "".join(character for character in text if not character.isspace())

    assert reconstructed == expected


def test_negation_scope_stops_at_contrast_conjunction():
    tokens = preprocess_text("The screen is not bright, but the keyboard is good.")
    by_text = {token.text: token for token in tokens}

    assert by_text["bright"].negated is True
    assert by_text["keyboard"].negated is False
    assert by_text["good"].negated is False


def test_aspect_offsets_create_bio_labels():
    text = "The battery life is poor."
    tokens = preprocess_text(text)
    aspects = [{"start": 4, "end": 16}]

    labels = align_aspects_to_bio(tokens, aspects)
    token_labels = dict(zip([token.text for token in tokens], labels))

    assert token_labels["battery"] == "B-ASPECT"
    assert token_labels["life"] == "I-ASPECT"
    assert token_labels["poor"] == "O"
