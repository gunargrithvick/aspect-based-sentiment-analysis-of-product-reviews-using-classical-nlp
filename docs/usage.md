# Usage and Reproducibility

## Environment setup

From the project root:

```bash
python -m venv .venv
```

Activate the environment and install dependencies using the instructions in the root `README.md`.

## Dataset and preprocessing commands

From the project root, run:

```bash
python scripts/prepare_laptop_dataset.py
python scripts/explore_laptop_dataset.py
python scripts/prepare_preprocessed_data.py
python scripts/train_baseline.py
python scripts/train_improved_models.py
python scripts/evaluate_and_improve.py
python scripts/verify_system.py
```

These commands parse the raw XML, create exploratory-analysis outputs, generate token-level model inputs, and train both the baseline and improved models.

## Browser interface

After the runtime models exist, start the FastAPI application with:

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` for the local HTML/CSS/JavaScript interface. The Vercel deployment setup is documented in the [frontend guide](frontend_guide.md).

## NLP resources

The current preprocessing implementation does not require a downloaded NLTK model or external language resource. If a later experiment uses NLTK tokenizers, taggers, or WordNet resources, download only the specific resources required by the code and record their names and versions in the final report.

## Running the development workflow

The development commands are documented in the root `README.md`. Any notebook used during development should record:

- Dataset path
- Random seed
- Preprocessing choices
- Feature settings
- Model parameters
- Evaluation results

## Reproducibility checklist

- Use a fixed random seed.
- Keep the official test set untouched during tuning.
- Record Python and package versions.
- Preserve the raw dataset separately from processed data.
- Save model settings with trained model artifacts.
- Record warnings and excluded records.
- Update the changelog when the workflow changes.
