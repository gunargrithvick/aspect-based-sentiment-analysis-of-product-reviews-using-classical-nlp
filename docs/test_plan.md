# Test Plan

## Automated tests

Run:

```bash
python -m pytest -q
```

Tests cover:

- XML annotation parsing
- Character-offset validation
- Tokenization and offset preservation
- Negation scope
- BIO aspect alignment
- Baseline aspect matching
- CRF feature construction and training

## Manual system checks

Before a demonstration, test the runtime interface with:

1. A review containing two positive and negative aspects.
2. A review containing a multiword aspect such as `battery life`.
3. A review containing negation.
4. A review with no obvious product aspect.
5. Empty input, which should produce a clear validation error.

## Acceptance criteria

The standalone system is acceptable when it loads the trained models, returns structured aspect-sentiment results, preserves character positions, handles multiple aspects, and passes the automated test suite.
