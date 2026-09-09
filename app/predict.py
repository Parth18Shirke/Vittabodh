"""
Prediction service — loads the trained DNN + preprocessors once at process
start (not per-request) so inference stays fast, and exposes a single
`predict()` call used by the Flask routes.

Improvements over the original:
  - APPROVAL_THRESHOLD is configurable via env var (default 0.50).
  - Unseen categorical values are logged as WARNING instead of silently
    falling back, making data-quality problems visible in the logs.
  - Optional Platt-scaling calibrator (probability_calibrator.pkl) is
    applied when present, so the confidence value is truly calibrated.
  - Drift detection warns when input features are far from the training
    distribution, flagging potentially unreliable predictions.
"""
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .drift_detector import check_drift
from .elrs import compute_elrs
from .explain import explain_prediction

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
APPROVAL_THRESHOLD = float(os.environ.get("APPROVAL_THRESHOLD", "0.50"))

_model = None
_scaler = None
_encoders = None
_feature_order = None
_calibrator = None   # Optional sklearn Platt scaler; may be None


def load_artifacts():
    global _model, _scaler, _encoders, _feature_order, _calibrator
    if _model is not None:
        return
    t0 = time.time()
    _model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "vittabodh_dnn.keras"))
    _scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
    _encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    _feature_order = joblib.load(os.path.join(MODEL_DIR, "feature_order.pkl"))

    calibrator_path = os.path.join(MODEL_DIR, "probability_calibrator.pkl")
    if os.path.exists(calibrator_path):
        _calibrator = joblib.load(calibrator_path)
        logger.info("Probability calibrator (Platt scaling) loaded")
    else:
        logger.info(
            "No probability_calibrator.pkl found — using raw sigmoid probabilities. "
            "Re-run train/train_model.py to enable calibration."
        )

    # Warm up the graph so the first real request is not slow
    dummy = np.zeros((1, len(_feature_order)), dtype=np.float32)
    _model.predict(dummy, verbose=0)
    logger.info(f"Model + preprocessors loaded and warmed up in {time.time() - t0:.2f}s")


def _encode_applicant(applicant: dict) -> np.ndarray:
    row = []
    for col in _feature_order:
        if col in _encoders:
            le = _encoders[col]
            val = str(applicant[col])
            if val not in le.classes_:
                logger.warning(
                    f"Unseen category '{val}' for feature '{col}' — falling back to "
                    f"'{le.classes_[0]}'. Consider retraining with updated data."
                )
                val = le.classes_[0]
            row.append(le.transform([val])[0])
        else:
            row.append(float(applicant[col]))
    return np.array(row, dtype=np.float32).reshape(1, -1)


def predict(applicant: dict) -> dict:
    """
    applicant: dict with the 25 raw form fields.
    Returns: dict with decision, probability, ELRS breakdown, top
    contributing features, and drift info.
    """
    load_artifacts()
    t0 = time.time()

    # Drift check (advisory — prediction always proceeds)
    drifted = check_drift(applicant, _feature_order)

    raw_row = _encode_applicant(applicant)
    raw_df = pd.DataFrame(raw_row, columns=_feature_order)
    scaled_row = _scaler.transform(raw_df)

    raw_prob = float(_model.predict(scaled_row, verbose=0)[0][0])

    # Apply Platt calibration if available
    if _calibrator is not None:
        prob_approved = float(_calibrator.predict_proba([[raw_prob]])[0][1])
    else:
        prob_approved = raw_prob

    decision = "Approved" if prob_approved >= APPROVAL_THRESHOLD else "Rejected"
    confidence = prob_approved * 100 if decision == "Approved" else (1 - prob_approved) * 100

    elrs_result = compute_elrs(applicant).as_dict()
    top_features = explain_prediction(_model, scaled_row, _feature_order)

    elapsed = time.time() - t0
    logger.info(
        "Prediction complete",
        extra={"decision": decision, "latency_ms": round(elapsed * 1000, 1)},
    )

    return {
        "decision": decision,
        "approval_probability": round(prob_approved * 100, 2),
        "confidence": round(confidence, 2),
        "elrs": elrs_result,
        "top_features": top_features,
        "latency_ms": round(elapsed * 1000, 1),
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
    }
