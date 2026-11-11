"""Portable linear model for the single-target product sentiment fallback."""

import re
from itertools import pairwise


def word_features(text: str) -> dict[str, int]:
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))
    return dict.fromkeys([*words, *(f"{a} {b}" for a, b in pairwise(words))], 1)


def predict_product_sentiment(text: str, model: dict) -> str | None:
    weights = model["weights"]
    active = [weights[feature] for feature in word_features(text) if feature in weights]
    margin = model["intercept"] + sum(active)
    if len(active) < 2 or abs(margin) < 0.65:
        return None
    return "positive" if margin > 0 else "negative"
