# Methodology

## System objective

Given a product-review sentence, identify explicit aspect terms and classify the sentiment associated with each aspect.

## Processing pipeline

```text
Annotated reviews
    -> dataset parsing
    -> tokenization and normalization
    -> linguistic feature construction
    -> aspect extraction
    -> aspect-centered sentiment classification
    -> end-to-end evaluation
```

## Preprocessing decisions

- Tokenize with character offsets so aspect annotations remain aligned with the original sentence.
- Normalize case while retaining the original token text.
- Preserve punctuation because it helps identify negation-scope boundaries.
- Do not remove stopwords before model development; words such as `not`, `never`, and `but` carry sentiment information.
- Apply conservative suffix rules for baseline lemmatization.
- Add coarse deterministic POS features and negation-scope flags.

The current POS component is intentionally a transparent heuristic baseline. It will not be presented as equivalent to a high-accuracy statistical tagger. A later experiment may compare it with a classical NLTK POS tagger if the required resource is documented and installed.

## Baseline system

The baseline will use noun and noun-phrase patterns for aspect candidates. A sentiment lexicon, negation rules, and a simple TF-IDF classifier will provide reference results.

## Improved system

### Aspect extraction

Use a Conditional Random Field sequence-labeling model with BIO tags. Features may include:

- Current, previous, and next tokens
- Lowercase form
- Prefixes and suffixes
- POS tag
- Token length
- Capitalization and numeric indicators
- Local word context

### Sentiment classification

For each extracted aspect, construct a local context window and represent it with:

- TF-IDF word n-grams
- Character n-grams
- Negation indicators
- POS-based features
- Lexicon-based sentiment features

Train an SVM classifier and compare it with simpler traditional classifiers.

## Strict technology boundary

The system must not use LLMs, generative-AI services, transformers, pretrained sentence embeddings, pretrained word embeddings, or text-generation models. Classical POS taggers and sentiment lexicons are permitted because they are standard non-generative NLP resources.

## Implementation rule

Notebooks should explain and visualize experiments. Reusable parsing, preprocessing, feature, modeling, and evaluation logic belongs in `src/classical_absa/`. The preprocessing script writes token-level CSV and sequence-level JSONL files to `data/processed/`.
