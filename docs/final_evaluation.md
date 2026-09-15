# Evaluation and Error Analysis

## Evaluation design

The outer validation split contains 20% of the sentences and was not used to select hyperparameters. Hyperparameters were selected using an inner split of the outer training data:

- CRF: `c1` in `{0.0, 0.1}` and `c2` in `{0.1, 1.0}`
- SVM: `C` in `{0.25, 1.0, 4.0}`
- Random seeds: `42` for the outer split and `43` for inner tuning

## Model comparison

| System | Aspect F1 | Sentiment Macro-F1 | End-to-End F1 |
|---|---:|---:|---:|
| Lexicon + Logistic Regression baseline | 0.581 | 0.454 | 0.366 |
| Default CRF + Linear SVM | 0.702 | 0.419 | 0.429 |
| Tuned CRF + Linear SVM | 0.685 | 0.420 | 0.424 |
| Lexical-only CRF ablation | 0.599 | Not evaluated | Not evaluated |

## Interpretation

The full CRF feature set is substantially better than the lexical-only ablation, showing that contextual, POS, shape, and negation features contribute to aspect extraction.

The default CRF configuration is the strongest observed configuration on the outer validation split. The tuned configuration was selected correctly from inner data, but it did not outperform the default configuration on this particular held-out split. This result is retained rather than hidden; a single validation split cannot prove that one configuration will always generalize better.

The sentiment classifier remains the main weakness. Macro-F1 is reduced by the rare `conflict` class and by confusion between neutral and positive or negative examples.

## Error categories

The improved default system produced the following validation error categories:

| Error type | Count |
|---|---:|
| Missed aspect | 108 |
| Wrong sentiment | 104 |
| Boundary error | 59 |
| False-positive aspect | 28 |

Typical issues include long multiword aspects, unfamiliar product vocabulary, implicit sentiment, and conflict opinions such as a feature being wanted but having an undesirable side effect.

Representative examples and machine-readable details are stored in `outputs/error_analysis_summary.json` and `outputs/error_analysis.csv`.

## Current model decision

For the project report and standalone application, use the **default CRF + Linear SVM** as the primary improved system because it achieved the best observed outer-validation aspect and end-to-end F1 scores. Keep the tuned model as an engineering comparison, not as a separate deliverable.
