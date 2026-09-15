# Project Release Checklist

## System implementation

- [x] Dataset parser and validation
- [x] Deterministic preprocessing
- [x] Baseline aspect and sentiment models
- [x] Improved CRF and SVM models
- [x] Standalone command-line inference
- [x] Structured JSON output
- [x] Runtime model verification
- [x] FastAPI browser interface
- [x] HTML, CSS, and JavaScript frontend
- [x] Vercel deployment configuration
- [x] Runtime model artifacts included for deployment
- [x] Raw dataset excluded with documented reproduction instructions

## Testing

- [x] Unit tests pass
- [x] Python source compilation succeeds
- [x] Character offsets are checked
- [x] Empty input is checked
- [x] Known positive and negative examples are checked
- [x] Error analysis is generated

## Documentation

- [x] Root README
- [x] Project definition
- [x] Dataset guide
- [x] System requirements
- [x] System design
- [x] User guide
- [x] Test plan
- [x] Evaluation and limitations
- [x] Changelog
- [x] Frontend and deployment guide

## Final commands

From the project root:

```bash
python scripts/build_system.py
python scripts/analyze_review.py --text "The battery life is terrible."
```

The build command regenerates processed data, models, evaluation outputs, and verification results.

## Submission items

- Source code under `src/`
- Build and utility scripts under `scripts/`
- Notebook or analysis materials under `notebooks/` if used
- Dataset source and handling description
- Evaluation tables under `reports/tables/`
- Technical documentation under `docs/`
- Presentation based on the implemented system
- Project report with results, limitations, and responsible-use notes
