# Baseline Results

## Experimental setup

The baseline uses:

- An exact-match aspect lexicon learned only from the training split
- A five-token aspect-centered context window
- TF-IDF word unigrams and bigrams
- Balanced Logistic Regression for sentiment classification
- Random seed `42`
- Sentence-level 80/20 training-validation split

The test data was not used for tuning. The current dataset contains only the annotated training XML, so these figures are validation results rather than official SemEval test results.

## Results

| Task | Metric | Result |
|---|---|---:|
| Aspect extraction | Precision | 0.504 |
| Aspect extraction | Recall | 0.684 |
| Aspect extraction | F1-score | 0.581 |
| Sentiment on gold aspects | Accuracy | 0.584 |
| Sentiment on gold aspects | Macro-F1 | 0.454 |
| End-to-end aspect and sentiment | Precision | 0.318 |
| End-to-end aspect and sentiment | Recall | 0.432 |
| End-to-end aspect and sentiment | F1-score | 0.366 |

## Interpretation

The lexicon baseline finds many known aspects but cannot identify unseen aspect terms, which explains its limited precision and recall. Sentiment classification is stronger for positive and negative examples than for the less frequent conflict class. End-to-end performance is lower because an error in either aspect extraction or sentiment prediction makes the complete prediction incorrect.

The baseline is a reference point. The improved CRF and SVM system should be compared against these exact metrics using the same split and evaluation definitions.
