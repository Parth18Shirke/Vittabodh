"""
Prediction history using Firestore as primary store,
with an in-process memory dict as a fast-access fallback.

This replaces the old per-process _RESULT_CACHE in routes.py, which
broke when Gunicorn spawned multiple workers (each worker had its own
dict, so PDF downloads that hit a different worker returned 404).
Firestore is process-agnostic: any worker can read any token.
"""
import logging
import time
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# -- In-memory fallback ----------------------------------------------------
_MEMORY_CACHE: dict = {}
_CACHE_TTL_S = 60 * 60  # 1 hour

_firestore_available: bool | None = None  # None = untested, True/False = known


def _get_db():
    global _firestore_available
    if _firestore_available is False:
        return None
    try:
        from firebase_admin import firestore
        db = firestore.client()
        _firestore_available = True
        return db
    except Exception as exc:
        if _firestore_available is None:
            logger.warning(
                f"Firestore unavailable — history falls back to in-memory store: {exc}"
            )
        _firestore_available = False
        return None


def _cleanup_memory():
    now = time.time()
    stale = [k for k, v in _MEMORY_CACHE.items() if now - v.get("ts", 0) > _CACHE_TTL_S]
    for k in stale:
        _MEMORY_CACHE.pop(k, None)


# -- Public API ------------------------------------------------------------

def save_prediction(uid: str, applicant: dict, result: dict, explanation: str, token: str | None = None) -> str:
    """Persist a prediction. Returns the cache token."""
    if token is None:
        token = uuid.uuid4().hex

    data = {
        "uid": uid,
        "token": token,
        "timestamp": datetime.utcnow().isoformat(),
        "ts": time.time(),
        "decision": result["decision"],
        "approval_probability": result["approval_probability"],
        "elrs_total": result["elrs"]["total"],
        "elrs_band": result["elrs"]["band"],
        "confidence": result["confidence"],
        "explanation_source": result.get("explanation_source", "fallback"),
        "applicant": applicant,
        "result": result,
        "explanation": explanation,
    }

    db = _get_db()
    if db:
        try:
            db.collection("predictions").document(token).set(data)
        except Exception as exc:
            logger.warning(f"Firestore write failed for token {token}: {exc}")

    _MEMORY_CACHE[token] = data
    _cleanup_memory()
    return token


def get_by_token(token: str) -> dict | None:
    """Fetch a prediction by token. Fast memory path, Firestore fallback."""
    entry = _MEMORY_CACHE.get(token)
    if entry and (time.time() - entry.get("ts", 0)) < _CACHE_TTL_S:
        return entry

    db = _get_db()
    if db:
        try:
            doc = db.collection("predictions").document(token).get()
            if doc.exists:
                data = doc.to_dict()
                _MEMORY_CACHE[token] = data
                return data
        except Exception as exc:
            logger.warning(f"Firestore read failed for token {token}: {exc}")

    return None


def get_history(uid: str, limit: int = 20) -> list:
    """Return recent predictions for a user, newest first."""
    db = _get_db()
    if db:
        try:
            docs = (
                db.collection("predictions")
                .where("uid", "==", uid)
                .order_by("ts", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            return [d.to_dict() for d in docs]
        except Exception as exc:
            logger.warning(f"Firestore history query failed for uid {uid}: {exc}")

    # Memory fallback
    entries = [v for v in _MEMORY_CACHE.values() if v.get("uid") == uid]
    entries.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return entries[:limit]


def get_admin_stats(limit: int = 200) -> dict:
    """Aggregate stats for the admin panel."""
    db = _get_db()
    all_preds: list = []

    if db:
        try:
            docs = (
                db.collection("predictions")
                .order_by("ts", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            all_preds = [d.to_dict() for d in docs]
        except Exception as exc:
            logger.warning(f"Firestore admin stats query failed: {exc}")

    if not all_preds:
        all_preds = sorted(_MEMORY_CACHE.values(), key=lambda x: x.get("ts", 0), reverse=True)[:limit]

    total = len(all_preds)
    if total == 0:
        return {
            "total": 0, "approved": 0, "rejected": 0,
            "approval_rate": 0, "band_counts": {}, "daily_counts": {}, "recent": [],
        }

    approved = sum(1 for p in all_preds if p.get("decision") == "Approved")
    band_counts: dict = {}
    for p in all_preds:
        band = p.get("elrs_band", "Unknown")
        band_counts[band] = band_counts.get(band, 0) + 1

    # Daily counts for last 7 days
    daily: dict = {}
    for i in range(7):
        day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        daily[day] = 0
    for p in all_preds:
        day = (p.get("timestamp") or "")[:10]
        if day in daily:
            daily[day] += 1

    return {
        "total": total,
        "approved": approved,
        "rejected": total - approved,
        "approval_rate": round(approved / total * 100, 1),
        "band_counts": band_counts,
        "daily_counts": dict(sorted(daily.items())),
        "recent": all_preds[:10],
    }
