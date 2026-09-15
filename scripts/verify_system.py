"""Verify the trained standalone NLP system and its runtime contract."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.inference_pipeline import AspectBasedSentimentAnalyzer  # noqa: E402


def check_result_spans(text: str, results: list[dict]) -> None:
    for result in results:
        start = result["start"]
        end = result["end"]
        if text[start:end] != result["aspect"]:
            raise AssertionError(
                f"Invalid span for {result!r}: text[{start}:{end}] does not match the aspect."
            )
        if result["sentiment"] not in {"positive", "negative", "neutral", "conflict"}:
            raise AssertionError(f"Unexpected sentiment label: {result['sentiment']}")


def main() -> None:
    analyzer = AspectBasedSentimentAnalyzer.from_model_directory(PROJECT_ROOT / "models")
    test_cases = {
        "The battery life is terrible.": "negative",
        "The screen is excellent.": "positive",
    }
    for text, expected_sentiment in test_cases.items():
        results = analyzer.analyze(text)
        if not results:
            raise AssertionError(f"No aspect was detected for required test input: {text}")
        check_result_spans(text, results)
        if results[0]["sentiment"] != expected_sentiment:
            raise AssertionError(
                f"Unexpected result for {text!r}: expected {expected_sentiment}, got {results}"
            )

    try:
        analyzer.analyze("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("Empty input should raise ValueError.")

    print("Standalone system verification passed.")
    print(f"Verified model directory: {PROJECT_ROOT / 'models'}")
    print(f"Verified inputs: {len(test_cases)}")


if __name__ == "__main__":
    main()
