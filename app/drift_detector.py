"""
Lightweight feature-drift detection.

During training, feature_stats.json is saved alongside the model
(mean + std for every numeric feature). At inference time we check
whether any feature is more than 3.5 standard deviations from the
training mean and log a warning. The prediction still proceeds —
the detector is advisory only.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_stats.json")
_stats: dict | None = None


def _load_stats() -> dict:
    global _stats
    if _stats is not None:
        return _stats
    try:
        with open(_STATS_PATH) as f:
            _stats = json.load(f)
        logger.info("feature_stats.json loaded — drift detection active")
    except FileNotFoundError:
        _stats = {}
        logger.info(
            "feature_stats.json not found — drift detection disabled. "
            "Re-run train/train_model.py to enable it."
        )
    except Exception as exc:
        _stats = {}
        logger.warning(f"Could not load feature_stats.json: {exc}")
    return _stats


def check_drift(applicant: dict, feature_order: list, threshold: float = 3.5) -> list[dict]:
    """
    Returns a list of dicts for features that deviate more than
    `threshold` standard deviations from the training distribution.
    Empty list means no drift detected.
    """
    stats = _load_stats()
    if not stats:
        return []

    drifted: list[dict] = []
    for feat in feature_order:
        if feat not in stats:
            continue
        raw = applicant.get(feat)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue

        mean = stats[feat]["mean"]
        std = stats[feat]["std"]
        if std < 1e-8:
            continue

        z = abs((val - mean) / std)
        if z > threshold:
            drifted.append({
                "feature": feat,
                "z_score": round(z, 2),
                "value": val,
                "mean": round(mean, 2),
                "std": round(std, 2),
            })

    if drifted:
        names = [d["feature"] for d in drifted]
        logger.warning(
            f"Drift detected in {len(drifted)} feature(s): {names}. "
            "Prediction may be less reliable for out-of-distribution inputs."
        )
    return drifted
