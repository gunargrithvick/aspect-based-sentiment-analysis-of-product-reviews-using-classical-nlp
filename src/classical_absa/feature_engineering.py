"""Classical feature construction for aspect-level sentiment models."""

from __future__ import annotations

from .text_preprocessing import preprocess_text


def aspect_context(text: str, start: int, end: int, window: int = 5) -> str:
    """Build a local context string with the target aspect marked.

    A fixed token window prevents the sentiment classifier from treating every
    word in a long multi-aspect sentence as equally relevant.
    """

    if start < 0 or end <= start or end > len(text):
        raise ValueError("Aspect offsets must be valid character positions.")

    tokens = preprocess_text(text)
    overlapping = [
        index
        for index, token in enumerate(tokens)
        if token.start < end and token.end > start
    ]
    if not overlapping:
        raise ValueError("Aspect offsets do not overlap any token.")

    first = max(0, min(overlapping) - window)
    last = min(len(tokens), max(overlapping) + window + 1)
    context = []
    for index in range(first, last):
        if index in overlapping:
            context.append("__ASPECT__")
        else:
            token = tokens[index]
            value = token.lemma
            if token.negated:
                value = f"NEG_{value}"
            context.append(value)
    return " ".join(context)
