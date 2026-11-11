"""Conservative lexical features for English electronics and appliance reviews."""

from __future__ import annotations

import re

from nltk.metrics.distance import edit_distance

ASPECTS = {
    "electronics": "battery life|battery|screen|display|camera|charger|charging port|build quality|price",
    "computers": "laptop|keyboard|trackpad|touchpad|processor|ram|memory|ssd|hard drive|hinge|webcam"
                 "|fan|graphics card|power adapter|usb port",
    "phones": "phone|smartphone|touchscreen|cellular signal|fingerprint sensor|face unlock",
    "audio": "headphones|earbuds|speaker|speakers|microphone|sound quality|bass|noise cancellation|bluetooth",
    "appliances": "vacuum|suction|filter|washing machine|washer|dishwasher|dryer|refrigerator|fridge|freezer"
                  "|temperature control|water tank|dust bin",
}
ASPECT_DOMAINS = {term: domain for domain, terms in ASPECTS.items() for term in terms.split("|")}
ASPECT_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(ASPECT_DOMAINS, key=lambda term: (-len(term), term))) + r")\b",
    re.IGNORECASE,
)
OPINIONS = {
    **dict.fromkeys(
        ("excellent", "great", "amazing", "fantastic", "love", "clear", "crisp", "quiet", "powerful",
         "reliable", "good", "bright"),
        1,
    ),
    **dict.fromkeys(
        ("terrible", "awful", "bad", "hate", "weak", "poor", "muddy", "muffled", "tinny", "noisy", "loud",
         "broken", "useless", "disappointing", "slow", "flimsy", "leaking", "leaks", "overheats", "died", "dies",
         "freezes", "crashes", "broke", "failed"),
        -1,
    ),
}
SPELL_TARGETS = frozenset(
    word for term in ASPECT_DOMAINS for word in term.split() if len(word) >= 5
) | frozenset(word for word in OPINIONS if len(word) >= 5)
PROTECTED_WORDS = frozenset({
    "batter", "better", "caller", "carried", "charged", "clearer", "close", "clean", "cooling", "drawer",
    "drifts", "earth", "early", "faster", "fitting", "fridge", "frozen", "hearing", "higher", "quite", "quiet",
    "rarely", "really", "scary", "slower", "staring", "trackball", "louder", "longer", "lower", "morning",
    "mystery", "painter", "scanner", "shaken", "shower", "signal", "sound", "speaker", "steal", "still",
    "stopped", "stronger", "sudden", "supper", "terrible", "thinner", "vacuum", "walked", "warmer", "wiping",
})
WORD_PATTERN = re.compile(r"\b[^\W\d_]+(?:['’\-][^\W\d_]+)*\b")
FAILURE_PATTERN = re.compile(
    r"\b(?:stop(?:s|ped) working|does(?:n't| not) work|won't (?:work|start|turn on)|"
    r"will not (?:work|start|turn on))\b",
    re.IGNORECASE,
)
CLAUSE_BREAK = re.compile(r"[.!?;\n]+|,?\s+\b(?:but|however|whereas|although|yet)\b\s+", re.IGNORECASE)
IMPLICIT_CUES = (
    ("battery life", "electronics", "negative", r"\b(?:needs? (?:a )?recharg(?:e|ing)|runs? out of juice)\b"),
    ("battery life", "electronics", "positive", r"\blasts? (?:all day|through the day|for days)\b"),
    ("portability", "electronics", "positive", r"\b(?:easy to carry|light enough to carry)\b"),
    ("portability", "electronics", "negative", r"\b(?:too heavy|hard) to carry\b"),
    ("performance", "computers", "negative", r"\b(?:freezes?|lags?) (?:when|while|every time)\b"),
    ("sound quality", "audio", "negative", r"\b(?:sounds? (?:muffled|tinny)|can(?:not|'t) hear (?:a thing|anything))\b"),
    ("suction", "appliances", "negative", r"\b(?:leaves? (?:crumbs|dust|dirt) behind|won't pick up (?:crumbs|dust|dirt))\b"),
    ("temperature control", "appliances", "negative", r"\b(?:won't stay cold|doesn't keep (?:food|drinks) cold)\b"),
)
SARCASM_PATTERN = re.compile(
    r"\b(?:yeah right|as if that helps)\b"
    r"|^\s*(?:oh[, ]+)?(?:great|brilliant|perfect|fantastic|wonderful)\b[,:\s]+"
    r"(?=[^.!?]*\b(?:died|dies|dead|broke|broken|failed|crashed|crashes|froze|freezes|overheats|leaks)\b)"
    r"(?![^.!?]*\b(?:not|never|no)\b)",
    re.IGNORECASE,
)


def normalize_spelling(text: str, known_words: frozenset[str]) -> tuple[str, list[dict], list[int]]:
    """Correct unique one-edit matches and map normalized boundaries to the original."""
    parts = []
    corrections = []
    boundaries = [0]
    cursor = 0
    replacements = {}
    for match in WORD_PATTERN.finditer(text):
        word = match.group()
        lower = word.lower()
        if len(word) < 5 or lower in known_words or not (word.islower() or word.istitle()):
            continue
        if lower not in replacements:
            candidates = [
                target for target in SPELL_TARGETS
                if abs(len(lower) - len(target)) <= 1 and edit_distance(lower, target, transpositions=True) == 1
            ]
            replacements[lower] = candidates[0] if len(candidates) == 1 else None
        target = replacements[lower]
        if target is None:
            continue
        start, end = match.span()
        replacement = target.capitalize() if word.istitle() else target
        parts.extend((text[cursor:start], replacement))
        boundaries.extend(range(cursor + 1, start + 1))
        boundaries.extend(start + index * (end - start) // len(replacement) for index in range(1, len(replacement) + 1))
        corrections.append({"original": word, "replacement": replacement, "start": start, "end": end})
        cursor = end
    parts.append(text[cursor:])
    boundaries.extend(range(cursor + 1, len(text) + 1))
    return "".join(parts), corrections, boundaries


def clause_ranges(text: str) -> list[tuple[int, int]]:
    start = 0
    ranges = []
    for match in CLAUSE_BREAK.finditer(text):
        ranges.append((start, match.start()))
        start = match.end()
    return [*ranges, (start, len(text))]


def lexical_sentiment(clause: str, center: float, targets: list[float]) -> str | None:
    """Assign opinion words to their nearest target within one clause."""
    failures = list(FAILURE_PATTERN.finditer(clause.replace("’", "'")))
    score = -sum(
        center == min(targets, key=lambda target: abs((match.start() + match.end()) / 2 - target))
        for match in failures
    )
    recent = []
    for match in WORD_PATTERN.finditer(clause.lower()):
        word = match.group().replace("’", "'")
        position = (match.start() + match.end()) / 2
        opinion = OPINIONS.get(word, 0)
        if opinion and center == min(targets, key=lambda target: abs(position - target)):
            negated = any(token in {"no", "not", "never"} or token.endswith("n't") for token in recent)
            score += -opinion if negated else opinion
        recent = [*recent[-2:], word]
    return "positive" if score > 0 else "negative" if score < 0 else None
