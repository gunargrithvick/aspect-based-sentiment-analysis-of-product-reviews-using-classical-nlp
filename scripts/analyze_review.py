"""Analyze one product review with the trained standalone NLP system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical_absa.inference_pipeline import AspectBasedSentimentAnalyzer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract product aspects and classify their sentiment."
    )
    parser.add_argument(
        "--text",
        help="Review text to analyze. If omitted, text is read interactively.",
    )
    args = parser.parse_args()
    text = args.text or input("Enter a product review: ").strip()

    analyzer = AspectBasedSentimentAnalyzer.from_model_directory(PROJECT_ROOT / "models")
    result = analyzer.analyze(text)
    print(json.dumps({"review": text, "results": result}, indent=2))


if __name__ == "__main__":
    main()
