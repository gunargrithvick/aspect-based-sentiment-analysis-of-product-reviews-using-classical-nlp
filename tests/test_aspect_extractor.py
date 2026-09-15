import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_absa.aspect_extractor import (
    CRFAspectExtractor,
    sentence_features,
)
from classical_absa.text_preprocessing import (
    align_aspects_to_bio,
    preprocess_text,
)


def test_crf_features_include_context_and_shape_information():
    tokens = preprocess_text("The battery is good.")
    features = sentence_features(tokens)

    assert features[1]["word.lower"] == "battery"
    assert features[1]["word.pos"]
    assert "+1:word.lower" in features[1]
    assert features[0]["BOS"] is True


def test_crf_can_train_and_return_one_label_per_token():
    texts = ["The battery is good.", "The screen is bright."]
    token_sequences = [preprocess_text(text) for text in texts]
    label_sequences = [
        align_aspects_to_bio(token_sequences[0], [{"start": 4, "end": 11}]),
        align_aspects_to_bio(token_sequences[1], [{"start": 4, "end": 10}]),
    ]

    extractor = CRFAspectExtractor().fit(token_sequences, label_sequences)
    predicted = extractor.predict_labels(token_sequences[0])

    assert len(predicted) == len(token_sequences[0])
