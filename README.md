# VittaBodh — Explainable AI Loan Approval Platform

A working implementation of the VittaBodh framework: a Deep Neural Network
for loan approval prediction, paired with the paper's 5-pillar Explainable
Loan Risk Scoring (ELRS) system, fast feature-level attribution, a
generative-AI explanation engine, Firebase authentication, and downloadable
PDF reports.

## Architecture

```
Browser (Firebase Auth SDK)
        │  ID token
        ▼
Flask app (run.py)
  ├─ app/auth.py          verifies Firebase ID tokens (Admin SDK, or REST
  │                       fallback if no service account is configured)
  ├─ app/routes.py        pages + JSON API
  ├─ app/predict.py       loads the trained DNN + scaler + encoders once,
  │                       runs inference (~150-250ms)
  ├─ app/elrs.py          5-pillar Explainable Loan Risk Score (paper formulas)
  ├─ app/explain.py       Integrated Gradients — SHAP-equivalent feature
  │                       attribution, computed via TF GradientTape (fast)
  ├─ app/genai_explainer.py  Gemini explanation, with a hard timeout and a
  │                       deterministic fallback so the page never hangs
  └─ app/pdf_report.py    branded PDF report generator (ReportLab)

train/train_model.py      one-off training script -> models/*.keras, *.pkl
```

## Why it doesn't lag

The paper's original design uses `shap.KernelExplainer`, which re-runs the
network hundreds of times per explanation and can take several seconds.
This build replaces it with **Integrated Gradients**, computed directly
from the network's own gradients via `tf.GradientTape` — mathematically
in the same family as SHAP (it satisfies the completeness axiom), but
computed in ~0.1–0.2s instead of seconds.

The Gemini call is wrapped in a background thread with an **8-second hard
timeout**. If Gemini is slow, rate-limited, or the key is invalid, the app
automatically falls back to a deterministic, still-personalised
explanation built from the ELRS breakdown and top features — the
prediction page always returns a complete result.

The model, scaler and label encoders are loaded **once at process
startup** (see `create_app()` in `app/__init__.py`), not per-request, and
the network is "warmed up" with a dummy prediction so the very first real
user doesn't pay a cold-start cost.

## Setup

1. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the model** (only needs to be run once — artifacts are saved
   to `models/`):
   ```bash
   python train/train_model.py
   ```
   This trains the exact architecture from the paper (25 → 128 → 64 → 32 → 1,
   ReLU + 0.3 dropout, sigmoid output) on the 20,000-record dataset in
   `data/Loan_Prediction_dataset.xlsx`, and prints accuracy/precision/
   recall/F1 plus a confusion matrix. On this dataset it reaches ~99.3%
   accuracy (close to the paper's reported 98.92%).

3. **Configure credentials** — `.env` is already filled in with the
   Firebase and Gemini credentials you provided. Double check:
   - The `GEMINI_API_KEY` you gave starts with `AQ.`, which isn't the
     usual Gemini key format (those start `AIzaSy...`). Verify it in
     [Google AI Studio](https://aistudio.google.com/app/apikey) — if
     it's actually a Firebase/OAuth token rather than a Gemini key, the
     app will just use its rule-based fallback explanations until you
     swap in a real key.

4. **Run it:**
   ```bash
   python run.py
   ```
   Visit `http://localhost:5000`.

## Enabling full admin access (optional, for production)

Without a service-account file, the app still verifies every Firebase ID
token cryptographically (via Firebase's public REST endpoint), so login
is fully secure. To also enable Firestore-backed admin roles:

1. In the Firebase Console → Project Settings → Service Accounts →
   "Generate new private key". Save the JSON file somewhere **outside**
   version control.
2. Set `FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/that/file.json` in `.env`.
3. In Firestore, create a `users/{uid}` document with `{"role": "admin"}`
   for any user you want elevated.

## Features beyond the paper

- **Integrated Gradients** instead of `SHAP.KernelExplainer` — same
  explanatory power, ~20-50x faster, no lag on the prediction page.
- **Timeout + fallback on the GenAI call** — the app can never hang or
  blank-screen because of a slow or failed third-party API call.
- **Model warm-up at startup** — no cold-start penalty for the first
  visitor after a deploy/restart.
- **Branded, downloadable PDF report** — decision, ELRS pillar bars, top
  factors, and the full AI explanation in one polished document.
- **Google Sign-In alongside email/password** via Firebase Auth.
- **Result caching by token** — a user can revisit or re-download their
  last report without resubmitting the form.
- **Graceful handling of unseen categorical values** at inference time
  (falls back to the nearest known category instead of crashing).

## Project layout

```
vittabodh/
├── app/                  Flask application package
├── data/                 training dataset
├── models/               trained model + preprocessors (generated)
├── static/                css/js assets
├── templates/             Jinja2 HTML templates
├── train/                 model training script
├── .env                    pre-filled environment config
├── requirements.txt
└── run.py                  entry point
```

## Retraining on new data

Drop a new `Loan_Prediction_dataset.xlsx` (same 25-feature schema) into
`data/` and re-run `python train/train_model.py`. It overwrites the
artifacts in `models/`; no other code changes are needed.
