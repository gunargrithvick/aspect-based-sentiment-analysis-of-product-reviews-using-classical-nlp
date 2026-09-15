# System Integration Verification

## Verification command

Run the following after generating the improved model files:

```bash
python scripts/verify_system.py
```

The verification checks that:

- Both runtime model files can be loaded.
- The system accepts non-empty review text.
- At least one known aspect is detected for controlled sample inputs.
- Returned character offsets match the returned aspect text.
- Returned sentiment labels are valid.
- Empty input raises a clear validation error.

## Verified sample behavior

Input:

```text
The battery life is terrible.
```

Output:

```text
battery life -> negative
```

Input:

```text
The screen is excellent.
```

Output:

```text
screen -> positive
```

These are controlled verification examples, not a claim that every review will be classified correctly. The validation metrics and known failure cases remain documented in `docs/final_evaluation.md`.
