"""Classical CRF-based aspect-term extraction."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn_crfsuite import CRF

from .text_preprocessing import Token, preprocess_text


def token_features(
    tokens: Sequence[Token], index: int, feature_profile: str = "full"
) -> dict[str, object]:
    """Build transparent lexical and contextual features for one token."""

    token = tokens[index]
    if feature_profile not in {"full", "lexical_only"}:
        raise ValueError(f"Unknown feature profile: {feature_profile}")

    features: dict[str, object] = {
        "bias": 1.0,
        "word.lower": token.normalized,
        "word.lemma": token.lemma,
    }
    if feature_profile == "lexical_only":
        return features

    features.update(
        {
            "word.pos": token.pos,
            "word.prefix2": token.normalized[:2],
            "word.suffix2": token.normalized[-2:],
            "word.prefix3": token.normalized[:3],
            "word.suffix3": token.normalized[-3:],
            "word.isupper": token.text.isupper(),
            "word.istitle": token.text.istitle(),
            "word.isdigit": token.text.isdigit(),
            "word.length": len(token.text),
            "word.negated": token.negated,
        }
    )

    if index > 0:
        previous = tokens[index - 1]
        features.update(
            {
                "-1:word.lower": previous.normalized,
                "-1:word.pos": previous.pos,
                "-1:word.lemma": previous.lemma,
            }
        )
    else:
        features["BOS"] = True

    if index < len(tokens) - 1:
        following = tokens[index + 1]
        features.update(
            {
                "+1:word.lower": following.normalized,
                "+1:word.pos": following.pos,
                "+1:word.lemma": following.lemma,
            }
        )
    else:
        features["EOS"] = True
    return features


def sentence_features(
    tokens: Sequence[Token], feature_profile: str = "full"
) -> list[dict[str, object]]:
    """Build CRF features for one tokenized sentence."""

    return [
        token_features(tokens, index, feature_profile=feature_profile)
        for index in range(len(tokens))
    ]


class CRFAspectExtractor:
    """Extract aspect terms with BIO sequence labeling."""

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        feature_profile: str = "full",
    ) -> None:
        self.feature_profile = feature_profile
        self.model = CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def fit(
        self, token_sequences: Sequence[Sequence[Token]], label_sequences: Sequence[Sequence[str]]
    ) -> CRFAspectExtractor:
        if len(token_sequences) != len(label_sequences):
            raise ValueError("Token and label sequence counts must match.")
        features = [
            sentence_features(tokens, feature_profile=self.feature_profile)
            for tokens in token_sequences
        ]
        self.model.fit(features, list(label_sequences))
        return self

    def predict_labels(self, tokens: Sequence[Token]) -> list[str]:
        return list(
            self.model.predict(
                [sentence_features(tokens, feature_profile=self.feature_profile)]
            )[0]
        )

    def predict(self, text: str) -> list[dict]:
        """Extract non-overlapping aspect spans from a raw sentence."""

        tokens = preprocess_text(text)
        labels = self.predict_labels(tokens)
        aspects: list[dict] = []
        current_indices: list[int] = []

        def flush() -> None:
            if not current_indices:
                return
            first = tokens[current_indices[0]]
            last = tokens[current_indices[-1]]
            aspects.append(
                {
                    "aspect_term": text[first.start : last.end],
                    "start": first.start,
                    "end": last.end,
                }
            )
            current_indices.clear()

        for index, label in enumerate(labels):
            if label == "B-ASPECT":
                flush()
                current_indices.append(index)
            elif label == "I-ASPECT" and current_indices:
                current_indices.append(index)
            else:
                flush()
        flush()
        return aspects
