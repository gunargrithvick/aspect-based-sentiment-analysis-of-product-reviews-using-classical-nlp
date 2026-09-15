# Aspect-Based Sentiment Analysis of Product Reviews Using Classical NLP

## 1. Project overview

This project implements a standalone NLP system that identifies product aspects in English laptop reviews and predicts the sentiment associated with each aspect.

Example:

```text
The battery life is terrible.
```

System output:

```text
battery life -> negative
```

The system uses classical NLP and traditional machine learning only. It does not use LLMs, generative AI, transformers, pretrained embeddings, or external AI services. A FastAPI application and HTML/CSS/JavaScript browser frontend provide a local user interface for the NLP pipeline.

## 2. Objectives

- Extract explicitly mentioned product aspects.
- Assign positive, negative, neutral, or conflict sentiment to each aspect.
- Preserve the character position of each detected aspect.
- Provide a standalone command-line interface.
- Provide a local browser interface backed by FastAPI.
- Compare a simple baseline with an improved classical NLP implementation.
- Validate the system using component-level and end-to-end metrics.

## 3. Dataset

The system uses the SemEval-2014 Task 4 Laptop Reviews training dataset. The parsed data contains:

- 3,045 sentences
- 2,358 aspect annotations
- 955 unique normalized aspect terms
- 987 positive annotations
- 866 negative annotations
- 460 neutral annotations
- 45 conflict annotations

The original XML is stored unchanged in `data/raw/Laptop_Train_v2.xml`. Generated files are stored separately under `data/processed/`.

## 4. System design

```text
Review text
    -> tokenization and normalization
    -> CRF aspect extraction
    -> aspect-centered context creation
    -> TF-IDF transformation
    -> Linear SVM sentiment classification
    -> structured JSON output
```

The main implementation is under `src/classical_absa/`. The command-line interface is `scripts/analyze_review.py`.

## 5. Preprocessing

The preprocessing system performs:

- Deterministic tokenization
- Character-offset preservation
- Case normalization
- Conservative rule-based lemmatization
- Coarse POS features
- Negation-scope detection
- BIO alignment of aspect annotations

Stopwords are not removed because words such as `not`, `never`, and `but` can change sentiment.

## 6. Models

### Baseline

- Exact aspect lexicon learned from the training split
- TF-IDF word unigrams and bigrams
- Balanced Logistic Regression sentiment classifier

### Improved system

- CRF sequence labeling for aspect extraction
- Lexical, contextual, word-shape, POS, and negation features
- TF-IDF word unigrams and bigrams
- Balanced Linear SVM sentiment classifier

## 7. Evaluation results

The evaluation uses a sentence-level 80/20 split with random seed `42`. Hyperparameter checks use an inner split and do not use the outer validation data for tuning.

| System | Aspect F1 | Sentiment Macro-F1 | End-to-End F1 |
|---|---:|---:|---:|
| Lexicon + Logistic Regression baseline | 0.581 | 0.454 | 0.366 |
| Default CRF + Linear SVM | 0.702 | 0.419 | 0.429 |
| Tuned CRF + Linear SVM | 0.685 | 0.420 | 0.424 |

The default CRF + Linear SVM is the primary system because it achieved the best observed aspect extraction and end-to-end scores on the outer validation split.

## 8. Error analysis

The primary system produced:

- 108 missed aspects
- 104 wrong sentiment predictions
- 59 aspect-boundary errors
- 28 false-positive aspects

The most difficult cases involve long multiword aspects, unfamiliar product vocabulary, conflict opinions, implicit sentiment, and neutral examples.

## 9. Standalone usage

```bash
python scripts/build_system.py
python scripts/analyze_review.py --text "The battery life is terrible."
```

The output is JSON containing the review, aspect text, sentiment label, and character offsets.

## 10. Testing

The project contains automated tests for:

- Dataset parsing
- Offset validation
- Tokenization
- Negation handling
- BIO labeling
- Baseline extraction
- CRF features
- Inference output

The final test run passed all 9 tests.

## 11. Limitations

The system is limited to English product-review text and explicitly mentioned aspects. It may make mistakes with sarcasm, implicit aspects, spelling errors, unusual vocabulary, and multiple overlapping opinions. It should be used for coursework and review analysis, not high-impact decision-making.

## 12. Future improvements

- Add more domain-specific training data.
- Improve multiword aspect-boundary detection.
- Add a stronger classical sentiment lexicon.
- Investigate class balancing for the conflict label.
- Add a larger held-out test set.
- Compare additional traditional classifiers such as calibrated SVM and Naive Bayes.
