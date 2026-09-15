# Frontend Guide

The project includes a local FastAPI application with a separate HTML, CSS, and JavaScript frontend. The browser interface calls the existing classical NLP inference pipeline and does not add another language model.

## Start the application locally

From the project root, activate the project environment and run:

```powershell
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in a browser. The API documentation is available at `http://127.0.0.1:8000/docs`.

## What the interface provides

- Product-review text input
- Example reviews for quick testing
- Model availability status
- Aspect count and sentiment summary cards
- Highlighted aspect terms in the original review
- Aspect-level sentiment table with character positions
- JSON download of the analysis results

## Application structure

```text
app.py                         FastAPI application and API routes
frontend/templates/index.html  Browser page structure
public/static/css/styles.css   Browser styling
public/static/js/app.js        Browser interactions and API calls
public/static/favicon.svg      Browser tab icon
vercel.json                    Vercel function file configuration
```

The frontend calls:

- `GET /` for the HTML interface
- `GET /health` for model availability
- `POST /api/analyze` for review analysis

## Deploy to Vercel

The project is configured for Vercel's Python runtime. The root `app.py` exposes the FastAPI instance as `app`, `vercel.json` includes the trained runtime models and frontend files, and the required runtime dependencies are listed in `requirements.txt`. The raw training dataset is not needed by the deployed application.

Before deploying through Git, make sure the runtime model files `crf_aspect_extractor.joblib` and `svm_sentiment_classifier.joblib` in `models/` are committed. They are intentionally not ignored because the deployed application needs them at runtime.

Using the Vercel CLI:

```powershell
npm install -g vercel
vercel login
vercel
vercel --prod
```

Alternatively, import the repository from the Vercel dashboard. Vercel should detect the FastAPI application automatically. The deployed root URL serves the frontend, while `/docs` exposes the API documentation.

## Before starting

The trained model files must exist at:

```text
models/crf_aspect_extractor.joblib
models/svm_sentiment_classifier.joblib
```

If they are missing, rebuild the system first:

```powershell
python scripts/build_system.py
```

## Scope and limitations

This is a local or Vercel-hosted frontend for the NLP system. It supports English product reviews and works best when the review contains explicit aspects such as `battery`, `screen`, `keyboard`, or `camera`.
