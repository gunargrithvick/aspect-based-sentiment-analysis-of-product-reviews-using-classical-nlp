# System Requirements

## System purpose

The system analyzes English laptop or product-review text and returns the product aspects mentioned in the text together with the sentiment associated with each aspect.

## Functional requirements

1. Accept one non-empty review as text input.
2. Tokenize and normalize the review while preserving character offsets.
3. Extract explicitly mentioned product aspects.
4. Classify each extracted aspect as positive, negative, neutral, or conflict where supported by the trained model.
5. Return structured output containing the aspect, sentiment, and character positions.
6. Handle reviews with multiple aspects.
7. Return an empty result when no aspect is detected.
8. Provide clear errors when trained model files are missing.
9. Provide a local browser interface for entering reviews and viewing results.
10. Provide a JSON API endpoint for the browser interface.

## Non-functional requirements

- Use only classical NLP and traditional machine learning.
- Produce deterministic results for the same model and input.
- Keep raw data separate from generated data.
- Keep preprocessing and model logic reusable from Python modules.
- Store model and package settings needed for reproducibility.
- Provide a command-line way to run the trained system.
- Provide a local FastAPI server that can be deployed to Vercel.

## Explicit exclusions

The system does not generate text, call an LLM, use a transformer, use pretrained embeddings, process audio or images, or make product recommendations.
