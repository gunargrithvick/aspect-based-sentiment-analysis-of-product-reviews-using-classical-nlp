# Improved Classical NLP Results

## Experimental setup

The improved system uses the same fixed sentence-level 80/20 split as the baseline:

- Random seed: `42`
- Training sentences: `2,436`
- Validation sentences: `609`
- Training aspects: `1,918`
- Validation aspects: `440`

The aspect extractor is a CRF sequence-labeling model using lexical, contextual, word-shape, POS, and negation features. The sentiment classifier uses TF-IDF word unigrams and bigrams with a balanced Linear SVM.

## Comparison

| System | Aspect F1 | Sentiment accuracy | Sentiment macro-F1 | End-to-end F1 |
|---|---:|---:|---:|---:|
| Lexicon + Logistic Regression baseline | 0.581 | 0.584 | 0.454 | 0.366 |
| CRF + Linear SVM | 0.702 | 0.600 | 0.419 | 0.429 |

## Honest interpretation

The CRF substantially improves aspect extraction because it uses sequence context instead of matching only aspect terms observed in training. This also improves end-to-end performance.

The SVM slightly improves accuracy but does not improve macro-F1. Its conflict-class F1 is zero on this validation split, and neutral examples remain difficult. This means the improved system is not uniformly better on every metric. The final report should discuss class imbalance and compare additional feature settings before claiming an overall improvement.

The full machine-readable results are stored in `outputs/improved_results.json`, and the comparison table is stored in `reports/tables/model_comparison.csv`.
