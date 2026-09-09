"""
Generative AI Explanation Engine â€” powered by Google Gemini (google-genai SDK).

Converts the DNN's prediction + ELRS risk profile + Integrated-Gradients
top features into a personalised, human-readable explanation. Falls back
to a deterministic templated explanation if the Gemini call fails or times
out, so the app NEVER blank-screens or hangs waiting on a third-party API.
"""
import os
import threading

from google import genai
from google.genai import types

_client = None
_lock = threading.Lock()


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        _client = genai.Client(api_key=api_key)
        return _client


def _build_prompt(applicant, decision, probability, elrs, top_features):
    feat_lines = "\n".join(
        f"- {f['feature']}: {f['direction']} (impact score {f['impact']:.3f})"
        for f in top_features
    )
    return f"""You are a loan-decision explanation assistant for VittaBodh, an
explainable AI lending platform. Write a short, warm, professional
explanation (120-180 words) for the applicant below. Do not use markdown
headers. Structure it as: 1) a one-line summary of the decision, 2) the
main strengths, 3) the main risk factors, 4) one or two concrete,
actionable suggestions if rejected (or ways to get an even better rate if
approved). Be specific but do not repeat raw numbers awkwardly â€” write in
natural sentences.

Decision: {decision}
Model confidence: {probability:.1f}%
Overall ELRS Risk Score: {elrs['total']}/100 ({elrs['band']})
Pillar scores: Credit Health {elrs['credit_health']}/25, Income & Stability
{elrs['income_stability']}/25, Asset & Collateral {elrs['asset_collateral']}/20,
Debt Burden {elrs['debt_burden']}/15, Profile Risk {elrs['profile_risk']}/15

Top contributing factors:
{feat_lines}

Loan purpose: {applicant.get('LoanPurpose')}
Requested amount: {applicant.get('LoanAmount')}
"""


def _fallback_explanation(applicant, decision, probability, elrs, top_features):
    strengths = [f for f in top_features if f["impact"] > 0][:3]
    risks = [f for f in top_features if f["impact"] < 0][:3]

    lines = []
    if decision == "Approved":
        lines.append(
            f"Your loan application has been approved with a model confidence of "
            f"{probability:.1f}%. Your overall ELRS risk score is {elrs['total']}/100, "
            f"placing you in the '{elrs['band']}' category."
        )
    else:
        lines.append(
            f"Your loan application was not approved at this time (model confidence "
            f"{probability:.1f}% for rejection). Your overall ELRS risk score is "
            f"{elrs['total']}/100, placing you in the '{elrs['band']}' category."
        )

    if strengths:
        s = ", ".join(f["feature"] for f in strengths)
        lines.append(f"Your strongest positive factors were: {s}.")
    if risks:
        r = ", ".join(f["feature"] for f in risks)
        lines.append(f"The factors that weighed most against you were: {r}.")

    if decision != "Approved":
        lines.append(
            "To improve future eligibility, consider reducing existing EMI "
            "obligations relative to income, building a stronger savings or "
            "collateral cushion, and maintaining a clean repayment record over "
            "the next few months before reapplying."
        )
    else:
        lines.append(
            "Maintaining a low debt-to-income ratio and continuing timely "
            "repayments will help you qualify for even better terms in future."
        )

    return " ".join(lines)


def generate_explanation(applicant, decision, probability, elrs, top_features, timeout_s=8):
    """
    Tries Gemini with a hard timeout; always returns a usable explanation
    string, using the deterministic fallback if Gemini is slow, unavailable,
    or errors out. This guarantees the prediction page never hangs.
    """
    result = {"text": None}

    def _call():
        try:
            client = _get_client()
            if client is None:
                return
            prompt = _build_prompt(applicant, decision, probability, elrs, top_features)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=350, temperature=0.6,
                ),
            )
            text = (resp.text or "").strip()
            if text:
                result["text"] = text
        except Exception:
            pass

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if result["text"]:
        return result["text"], "gemini"

    return _fallback_explanation(applicant, decision, probability, elrs, top_features), "fallback"
