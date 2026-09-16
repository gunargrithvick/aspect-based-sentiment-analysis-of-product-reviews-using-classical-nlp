# Changelog

All notable project changes will be recorded here.

## Unreleased

### Added

- Defined the standalone classical NLP project scope.
- Added a reproducible project structure for data, notebooks, source code, tests, models, reports, and outputs.
- Added documentation conventions and dependency files.
- Standardized project naming and casing conventions.
- Added deterministic tokenization, aspect-offset alignment, lemmatization, POS heuristics, and negation features.
- Added token-level and sequence-level preprocessed dataset outputs.
- Added a training-split aspect-lexicon and TF-IDF Logistic Regression baseline with validation metrics.
- Added CRF aspect extraction, TF-IDF Linear SVM sentiment classification, and a model comparison report.
- Added inner-split hyperparameter evaluation, feature ablation, final error categorization, and validation analysis.
- Added the standalone command-line inference pipeline and system documentation.
- Added runtime integration verification for trained model artifacts and structured output.
- Added a reproducible full-system build command and submission documentation.
- Added the final project report and runnable system-analysis notebook.
- Added a local FastAPI application with a separate HTML, CSS, and JavaScript frontend.
- Added aspect highlighting, sentiment summaries, result tables, JSON export, health checks, and API documentation.
- Added Vercel deployment configuration and runtime documentation.

### Changed

- Aligned Python imports and type annotations with the CI lint toolchain.
- Added repository-wide Ruff configuration and pinned the development Ruff version for reproducible checks.
- Updated GitHub Actions to current checkout and Python setup action versions and extended linting to the full repository.
- Regenerated the committed runtime model artifacts through the complete build-and-verification workflow.
- Added explicit Author and License sections to the root README.
- Added README architecture, testing, and verified frontend screenshot sections.
