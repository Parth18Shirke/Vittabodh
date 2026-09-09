import io
import json
import logging
import os

from flask import (Blueprint, jsonify, render_template, request,
                   send_file, session)

from .auth import admin_required, login_required, verify_id_token
from .explain import generate_suggestions
from .genai_explainer import generate_explanation
from .history import get_admin_stats, get_by_token, get_history, save_prediction
from .limiter import limiter
from .pdf_report import build_pdf_report
from .predict import predict as run_prediction

logger = logging.getLogger(__name__)

web_bp = Blueprint("web", __name__)
api_bp = Blueprint("api", __name__)

# -- Field definitions ------------------------------------------------------

REQUIRED_FIELDS = [
    "Age", "Gender", "MaritalStatus", "Education", "Nationality", "Region",
    "EmploymentType", "IndustrySector", "YearsExperience", "MonthlyIncome",
    "MonthlyExpenses", "SavingsBalance", "InvestmentValue", "ExistingEMI",
    "CreditScore", "PastDefaults", "LatePayments", "ExistingLoans",
    "CurrentJobDuration", "Dependents", "LoanAmount", "LoanPurpose",
    "LoanTenure", "CollateralValue", "GuarantorAvailable",
]

NUMERIC_FIELDS = {
    "Age", "YearsExperience", "MonthlyIncome", "MonthlyExpenses", "SavingsBalance",
    "InvestmentValue", "ExistingEMI", "CreditScore", "PastDefaults", "LatePayments",
    "ExistingLoans", "CurrentJobDuration", "Dependents", "LoanAmount", "LoanTenure",
    "CollateralValue",
}

# Server-side range guard (HTML form has min/max but the JSON API has none)
FIELD_RANGES = {
    "Age":               (1,   120),
    "CreditScore":       (300, 900),
    "LoanTenure":        (1,   600),
    "PastDefaults":      (0,   50),
    "LatePayments":      (0,   100),
    "ExistingLoans":     (0,   50),
    "Dependents":        (0,   20),
    "YearsExperience":   (0,   60),
    "CurrentJobDuration":(0,   600),
    "MonthlyIncome":     (0,   1e9),
    "MonthlyExpenses":   (0,   1e9),
    "SavingsBalance":    (0,   1e12),
    "InvestmentValue":   (0,   1e12),
    "ExistingEMI":       (0,   1e9),
    "LoanAmount":        (1,   1e12),
    "CollateralValue":   (0,   1e12),
}


def _parse_and_validate(data: dict) -> tuple:
    """Returns (applicant_dict, None) or (None, error_str)."""
    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in (None, "")]
    if missing:
        return None, f"Missing fields: {', '.join(missing)}"

    applicant: dict = {}
    try:
        for f in REQUIRED_FIELDS:
            applicant[f] = float(data[f]) if f in NUMERIC_FIELDS else str(data[f])
    except (TypeError, ValueError) as exc:
        return None, f"Invalid field value: {exc}"

    for field, (lo, hi) in FIELD_RANGES.items():
        if field in applicant:
            val = applicant[field]
            if not (lo <= val <= hi):
                return None, (
                    f"'{field}' value {val} is outside the valid range "
                    f"[{lo}, {hi}]."
                )
    return applicant, None


# -- Firebase config helper -------------------------------------------------

def _firebase_config() -> dict:
    return {
        "apiKey":            os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain":        os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId":         os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket":     os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId":             os.environ.get("FIREBASE_APP_ID", ""),
        "measurementId":     os.environ.get("FIREBASE_MEASUREMENT_ID", ""),
    }


@web_bp.context_processor
def inject_firebase_config():
    return {"firebase_config": _firebase_config()}


# -- Page routes ------------------------------------------------------------

@web_bp.route("/")
def landing():
    return render_template("landing.html")


@web_bp.route("/login")
def login():
    return render_template("login.html", firebase_config=_firebase_config())


@web_bp.route("/dashboard")
@login_required
def dashboard():
    uid = session.get("uid")
    history = get_history(uid, limit=5)
    return render_template("dashboard.html", user=session.get("user"), history=history)


@web_bp.route("/apply")
@login_required
def apply():
    return render_template("apply.html", user=session.get("user"))


@web_bp.route("/history")
@login_required
def history_page():
    uid = session.get("uid")
    history = get_history(uid, limit=50)
    return render_template("history.html", user=session.get("user"), history=history)


@web_bp.route("/admin")
@admin_required
def admin():
    metrics_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "metrics.json"
    )
    model_metrics: dict = {}
    try:
        with open(metrics_path) as f:
            model_metrics = json.load(f)
    except Exception:
        pass
    return render_template(
        "admin.html", user=session.get("user"), model_metrics=model_metrics
    )


@web_bp.route("/result/<token>")
@login_required
def result_page(token):
    entry = get_by_token(token)
    if not entry:
        return render_template("expired.html"), 404
    suggestions = generate_suggestions(
        entry["result"].get("top_features", []),
        entry["result"].get("decision", ""),
    )
    return render_template(
        "result.html",
        token=token,
        applicant=entry["applicant"],
        result=entry["result"],
        explanation=entry["explanation"],
        suggestions=suggestions,
        user=session.get("user"),
    )


@web_bp.route("/logout")
def logout():
    session.clear()
    from flask import redirect, url_for
    return redirect(url_for("web.landing"))


# -- API: auth --------------------------------------------------------------

@api_bp.route("/session-login", methods=["POST"])
def session_login():
    data = request.get_json(silent=True) or {}
    id_token = data.get("idToken")
    if not id_token:
        return jsonify({"error": "Missing idToken"}), 400
    try:
        user = verify_id_token(id_token)
    except Exception as exc:
        return jsonify({"error": f"Token verification failed: {exc}"}), 401

    session["uid"] = user["uid"]
    session["user"] = user
    session["is_admin"] = _check_admin_role(user["uid"])
    session.permanent = True
    return jsonify({"ok": True, "user": user})


def _check_admin_role(uid: str) -> bool:
    """Best-effort Firestore role lookup; safe to fail."""
    try:
        from firebase_admin import firestore
        db = firestore.client()
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict().get("role") == "admin"
    except Exception:
        pass
    return False


# -- API: predict -----------------------------------------------------------

@api_bp.route("/predict", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def api_predict():
    data = request.get_json(silent=True) or {}
    applicant, error = _parse_and_validate(data)
    if error:
        return jsonify({"error": error}), 400

    result = run_prediction(applicant)

    explanation, source = generate_explanation(
        applicant, result["decision"], result["approval_probability"],
        result["elrs"], result["top_features"],
    )
    result["explanation_source"] = source

    uid = session.get("uid", "anonymous")
    token = save_prediction(uid, applicant, result, explanation)

    logger.info(
        "Prediction saved",
        extra={"uid": uid, "decision": result["decision"], "prediction_token": token},
    )
    return jsonify({"token": token, "result": result, "explanation": explanation})


# -- API: what-if simulator -------------------------------------------------

@api_bp.route("/whatif", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def api_whatif():
    """
    Takes a base token + field overrides dict, runs prediction without
    the Gemini call (fast path), and returns the new result.
    Used by the What-If Simulator sliders on the result page.
    """
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    overrides = data.get("overrides", {})

    if not token:
        return jsonify({"error": "Missing token"}), 400

    entry = get_by_token(token)
    if not entry:
        return jsonify({"error": "Result expired. Please re-submit the form."}), 404

    # Clone the stored applicant and apply overrides with range clamping
    applicant = dict(entry["applicant"])
    for field, value in overrides.items():
        if field in NUMERIC_FIELDS and field in FIELD_RANGES:
            try:
                lo, hi = FIELD_RANGES[field]
                applicant[field] = max(lo, min(hi, float(value)))
            except (TypeError, ValueError):
                pass

    result = run_prediction(applicant)
    return jsonify({"result": result})


# -- API: history -----------------------------------------------------------

@api_bp.route("/history")
@login_required
def api_history():
    uid = session.get("uid")
    full = get_history(uid, limit=50)
    slim = [
        {
            "token":               h.get("token"),
            "timestamp":           h.get("timestamp"),
            "decision":            h.get("decision"),
            "approval_probability":h.get("approval_probability"),
            "elrs_total":          h.get("elrs_total"),
            "elrs_band":           h.get("elrs_band"),
            "confidence":          h.get("confidence"),
        }
        for h in full
    ]
    return jsonify(slim)


# -- API: admin -------------------------------------------------------------

@api_bp.route("/admin/stats")
@admin_required
def api_admin_stats():
    stats = get_admin_stats(limit=200)
    return jsonify(stats)


# -- API: PDF report --------------------------------------------------------

@api_bp.route("/report/<token>.pdf")
@login_required
def api_report(token):
    entry = get_by_token(token)
    if not entry:
        return jsonify({"error": "Result expired. Please re-submit the form."}), 404

    user = session.get("user", {})
    suggestions = generate_suggestions(
        entry["result"].get("top_features", []),
        entry["result"].get("decision", ""),
    )
    pdf_bytes = build_pdf_report(
        entry["applicant"], entry["result"], entry["explanation"],
        applicant_name=user.get("name", "Applicant"),
        suggestions=suggestions,
    )
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf",
        as_attachment=True, download_name="VittaBodh_Loan_Report.pdf",
    )


@api_bp.route("/health")
def health():
    return jsonify({"status": "ok"})
