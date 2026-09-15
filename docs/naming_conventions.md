# Naming Conventions

The project uses a consistent naming policy so that files are easy to find and work consistently across Windows, macOS, and Linux.

## Folders

Use lowercase names with underscores only when a separator is needed:

```text
data/
data/raw/
src/classical_absa/
reports/figures/
```

## Python files

Use lowercase `snake_case` names:

```text
dataset_loader.py
text_preprocessing.py
feature_engineering.py
sentiment_classifier.py
```

## Documentation files

Use lowercase `snake_case` names for documents inside `docs/`:

```text
project_definition.md
dataset_guide.md
evaluation_plan.md
```

Repository-standard files may use their conventional uppercase names:

```text
README.md
CHANGELOG.md
CONTRIBUTING.md
```

## Data and output files

Generated files use lowercase `snake_case` names:

```text
laptop_review_sentences.csv
laptop_review_aspects.csv
dataset_validation.json
polarity_distribution.png
```

The raw file `Laptop_Train_v2.xml` intentionally preserves the casing supplied by the dataset distribution. Renaming an original source file could make provenance and reproducibility less clear.

## Notebook files

Notebook filenames use a numeric execution order followed by a lowercase descriptive name:

```text
01_dataset_exploration.ipynb
02_text_preprocessing_and_baseline.ipynb
03_crf_aspect_extraction.ipynb
04_svm_sentiment_classification.ipynb
05_end_to_end_evaluation.ipynb
```

The numbers indicate recommended order, not a directory or documentation category.
