# Dataset Guide

## Dataset

Use the SemEval-2014 Task 4 Laptop Reviews dataset. It contains review sentences annotated with aspect terms and their sentiment polarity.

The official task page is the SemEval-2014 Task 4 [Data and Tools page](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). The project currently uses `Laptop_Train_v2.xml`, obtained from the public [SemEval2014Task4 dataset mirror](https://huggingface.co/datasets/alexcadillon/SemEval2014Task4).

The official META-SHARE download endpoint required authentication in this environment. The mirror file was checked as XML and is used only as a convenient distribution of the original annotated training file. The dataset's original terms and conditions still apply.

## Storage rules

Place the unchanged downloaded files in:

```text
data/raw/
```

Do not edit the original files. Store converted annotations in:

```text
data/interim/
data/processed/
```

The raw dataset is ignored by version control because redistribution permissions may vary.

## Current data inventory

| File | Description |
|---|---|
| `data/raw/Laptop_Train_v2.xml` | Unchanged SemEval laptop-review training XML |
| `data/processed/laptop_review_sentences.csv` | One record per review sentence |
| `data/processed/laptop_review_aspects.csv` | One record per annotated aspect |
| `outputs/dataset_validation.json` | Parser and annotation validation summary |
| `outputs/dataset_summary.json` | Exploratory-analysis summary |

The current parsed dataset contains 3,045 sentences, 1,488 sentences with at least one aspect, 2,358 aspect annotations, and 955 unique normalized aspect terms. The polarity counts are 987 positive, 866 negative, 460 neutral, and 45 conflict.

## Expected converted records

Each processed aspect-level record should preserve enough information to reproduce the experiment:

```text
sentence_id
sentence_text
token_index
token
bio_tag
aspect_term
polarity
source_split
```

## Data checks

Before training, verify:

- Every aspect span matches the source sentence.
- BIO tags are valid and preserve multiword aspects.
- Sentiment labels are normalized consistently.
- Training and test records do not overlap.
- Class distributions are reported.
- Parsing warnings are saved for review rather than silently discarded.
