# Project Definition

## Project title

**Aspect-Based Sentiment Analysis of Product Reviews Using Classical NLP**

## Problem statement

Ordinary sentiment analysis decides whether an entire review is positive or negative. This project identifies individual product aspects and the sentiment associated with each aspect.

Example: `The camera is excellent, but the battery life is terrible.`

Expected output:

```text
camera      -> positive
battery life -> negative
```

## Objective

Build a standalone NLP system that accepts an English product review and extracts explicitly mentioned aspect terms, such as `battery`, `camera`, or `screen`, together with their sentiment polarity.

The system will perform:

1. Aspect-term extraction.
2. Aspect-level sentiment classification.

It also includes a lightweight FastAPI service and browser frontend so the implemented NLP pipeline can be used through a simple local or deployable interface.

## Initial dataset and domain

The first implementation will use the **SemEval-2014 Task 4 Laptop Reviews dataset**. The project will focus on explicitly mentioned aspects. Implicit opinions, such as `It is too expensive` where `price` is not stated, will be treated as a limitation or future extension.

## Input and output

The system will accept an English review sentence or a collection of annotated review sentences. It will return each detected aspect, its sentiment polarity, and optionally the source sentence.

```text
Input:  The display is bright, but the keyboard feels cheap.
Output: display -> positive; keyboard -> negative
```

## Proposed classical NLP approach

### Baseline

- POS-tag and noun-phrase-based aspect extraction rules
- Sentiment lexicon and negation rules
- TF-IDF with a traditional classifier for comparison

### Improved system

- CRF-based sequence labeling for aspect extraction
- Word and character n-gram features
- TF-IDF features for sentiment classification
- SVM-based aspect-level sentiment classification
- Negation, POS, and local-context features

## Project boundaries

The project will not use LLMs, generative-AI APIs, GPT, BERT, RoBERTa, other transformers, sentence-transformer models, pretrained embeddings, text generation, recommendation systems, speech, images, video, or multilingual processing in the first version. A local browser frontend is included only as a user interface for the NLP system; it does not add another NLP model or external service.

## Evaluation objectives

- Aspect extraction: precision, recall, F1-score, and exact-match performance.
- Sentiment classification: accuracy, precision, recall, macro-F1, and a confusion matrix.
- End-to-end evaluation: correct `(aspect, sentiment)` pairs.

## Success criteria

The completed system must preprocess the dataset, extract explicit product aspects, predict sentiment for each aspect, compare a baseline with an improved model, report suitable metrics, explain errors and limitations, and work without LLMs or generative AI.

## Final scope

The complete system will focus on **English laptop reviews, explicit aspect-term extraction, and aspect-level sentiment classification using rule-based NLP, CRF, TF-IDF, n-grams, and SVM**, with FastAPI and HTML/CSS/JavaScript providing the user interface.
