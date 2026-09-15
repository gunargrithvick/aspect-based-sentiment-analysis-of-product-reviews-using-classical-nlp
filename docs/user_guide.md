# User Guide

## Install dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Train the runtime models

The trained models must exist in `models/`. Generate them with:

```bash
python scripts/train_improved_models.py
```

## Analyze a review

Pass a review directly:

```bash
python scripts/analyze_review.py --text "The display is bright, but the keyboard feels cheap."
```

Or run without `--text` and enter the review when prompted:

```bash
python scripts/analyze_review.py
```

Example structured output:

```json
{
  "review": "The battery life is terrible.",
  "results": [
    {"aspect": "battery life", "sentiment": "negative", "start": 4, "end": 16}
  ]
}
```

The application does not generate a rewritten review or explanation. It only extracts text spans and assigns sentiment labels.

## Run the browser interface

Start the local FastAPI server:

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` to enter reviews through the HTML, CSS, and JavaScript frontend. The interface highlights predicted aspects, shows their sentiment, and supports JSON download. For deployment, see the [frontend guide](frontend_guide.md).
