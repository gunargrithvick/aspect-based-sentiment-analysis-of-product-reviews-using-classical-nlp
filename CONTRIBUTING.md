# Contribution Guidelines

## Project conventions

- Use Python and classical NLP methods only.
- Do not add LLMs, generative-AI APIs, transformers, pretrained embeddings, or text-generation dependencies.
- Use descriptive `snake_case` names for Python files and lowercase names for folders.
- Keep reusable logic in `src/classical_absa/`; use notebooks for experiments and explanations.
- Keep raw source data unchanged.
- Do not commit downloaded datasets, generated model files, notebook checkpoints, or evaluation outputs unless explicitly required for the final submission.

## Before submitting changes

Run the test suite:

```bash
python -m pytest
```

Check code style:

```bash
ruff check .
```

Update the relevant documentation and add an entry to `CHANGELOG.md` when behavior or project organization changes.
