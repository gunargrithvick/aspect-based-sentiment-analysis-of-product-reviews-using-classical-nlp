# Evaluation Plan

## Data split

Use the official training and test split supplied with the dataset. Create a validation split only from the training data, using a fixed random seed. Do not tune hyperparameters on the test set.

## Aspect extraction metrics

Report:

- Token-level precision, recall, and F1-score
- Exact-match precision, recall, and F1-score for complete aspect spans

Exact-match evaluation is important because predicting only part of `battery life` is not the same as predicting the complete aspect.

## Sentiment metrics

Report:

- Accuracy
- Per-class precision and recall
- Macro-F1 score
- Confusion matrix

Macro-F1 should be emphasized when the sentiment classes are imbalanced.

## End-to-end metric

Count a prediction as correct only when both the aspect span and its sentiment are correct:

```text
(battery life, negative)
```

## Required comparisons

Compare at least:

1. Rule-based aspect extraction with lexicon sentiment.
2. TF-IDF sentiment classification baseline.
3. CRF aspect extraction with SVM sentiment classification.

## Error analysis

Review representative errors involving:

- Negation
- Multiple aspects in one sentence
- Multiword aspects
- Implicit aspects
- Neutral or conflict labels
- Sarcasm and unusual product vocabulary

Save the final comparison tables in `reports/tables/` and figures in `reports/figures/`.
