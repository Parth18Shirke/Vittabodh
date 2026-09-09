"""
Feature-level explainability via Integrated Gradients.

The research paper uses SHAP's KernelExplainer for feature attribution.
KernelExplainer is model-agnostic but re-runs the network hundreds of times
per explanation, taking several seconds per prediction â€” unacceptable for a
live web form. Integrated Gradients gives an equivalent, theoretically
grounded per-feature attribution (it satisfies the same completeness axiom
SHAP is built on) using TensorFlow's GradientTape directly against the
network's own gradients, typically in ~0.1-0.2s. This keeps the app
responsive without sacrificing the "why" behind each prediction.
"""
import numpy as np
import tensorflow as tf

READABLE_LABELS = {
    "Age": "Age",
    "YearsExperience": "Years of Experience",
    "MonthlyIncome": "Monthly Income",
    "MonthlyExpenses": "Monthly Expenses",
    "SavingsBalance": "Savings Balance",
    "InvestmentValue": "Investment Value",
    "ExistingEMI": "Existing EMI",
    "CreditScore": "Credit Score",
    "PastDefaults": "Past Defaults",
    "LatePayments": "Late Payments",
    "ExistingLoans": "Existing Loans",
    "CurrentJobDuration": "Current Job Duration",
    "Dependents": "Dependents",
    "LoanAmount": "Loan Amount",
    "LoanTenure": "Loan Tenure",
    "CollateralValue": "Collateral Value",
    "Gender": "Gender",
    "MaritalStatus": "Marital Status",
    "Education": "Education",
    "Nationality": "Nationality",
    "Region": "Region",
    "EmploymentType": "Employment Type",
    "IndustrySector": "Industry Sector",
    "LoanPurpose": "Loan Purpose",
    "GuarantorAvailable": "Guarantor Availability",
}


def integrated_gradients(model, x, baseline=None, steps=32):
    """
    x: (1, n_features) scaled input
    baseline: (1, n_features), defaults to zero vector (dataset mean in
              scaled space, since StandardScaler centers features at 0)
    Returns: (n_features,) attribution array, sums ~ (f(x) - f(baseline))
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(x)
    else:
        baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps), (steps, 1))
    interpolated = baseline + alphas * (x - baseline)  # (steps, n_features)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)  # (steps, 1)
    grads = tape.gradient(preds, interpolated)  # (steps, n_features)

    avg_grads = tf.reduce_mean(grads, axis=0)  # (n_features,)
    attributions = (x[0] - baseline[0]) * avg_grads
    return attributions.numpy()


def explain_prediction(model, x_scaled, feature_order, top_n=6):
    """Returns top contributing features with signed attribution scores."""
    attributions = integrated_gradients(model, x_scaled)
    pairs = list(zip(feature_order, attributions))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)

    results = []
    for name, val in pairs[:top_n]:
        results.append({
            "feature": READABLE_LABELS.get(name, name),
            "raw_feature": name,
            "impact": float(val),
            "direction": "increases approval likelihood" if val > 0 else "decreases approval likelihood",
        })
    return results


# Suggestion templates keyed by raw feature name
_SUGGESTIONS = {
    "CreditScore": (
        "Improve your credit score by paying all dues on time. "
        "Check your credit report for errors and dispute any inaccuracies."
    ),
    "LatePayments": (
        "Reduce late payments — even one missed payment can lower your score significantly. "
        "Set up auto-debit for EMIs and credit card bills."
    ),
    "PastDefaults": (
        "A history of defaults is a major red flag. "
        "Build 6-12 months of clean repayment history before reapplying."
    ),
    "ExistingEMI": (
        "Your existing EMI burden is high relative to income. "
        "Consider closing smaller loans first to reduce your EMI-to-income ratio below 40%."
    ),
    "ExistingLoans": (
        "Having many active loans reduces your credit capacity. "
        "Consolidate or close at least one loan before applying for a new one."
    ),
    "MonthlyIncome": (
        "A higher income improves your debt-service coverage ratio. "
        "If possible, provide documentation of any additional income sources (freelance, rent, etc.)."
    ),
    "MonthlyExpenses": (
        "High monthly expenses reduce your net disposable income. "
        "Demonstrating lower expenses over 3-6 months can strengthen your profile."
    ),
    "SavingsBalance": (
        "Build a larger savings buffer — lenders view savings as an emergency cushion. "
        "Aim for at least 3 months of EMI in liquid savings."
    ),
    "InvestmentValue": (
        "Increasing investment assets signals financial stability. "
        "Regular SIP investments in mutual funds can build this over time."
    ),
    "CollateralValue": (
        "Offering collateral with a higher market value (property, FDs) "
        "significantly improves the Asset & Collateral score."
    ),
    "LoanAmount": (
        "Consider requesting a slightly lower loan amount. "
        "A lower loan-to-income ratio improves approval chances and gets better rates."
    ),
    "LoanTenure": (
        "Extending the loan tenure reduces monthly EMI burden "
        "and improves your debt-service coverage ratio."
    ),
    "CurrentJobDuration": (
        "Lenders prefer applicants with longer job stability (2+ years in current role). "
        "Avoid switching jobs just before reapplying."
    ),
    "YearsExperience": (
        "More years of work experience signals career stability. "
        "Document any certifications or promotions that demonstrate career growth."
    ),
    "GuarantorAvailable": (
        "Adding a creditworthy guarantor can significantly boost approval chances, "
        "especially if your own credit profile has weak spots."
    ),
    "Dependents": (
        "A high number of dependents increases perceived financial burden. "
        "Providing proof of a stable, growing income can offset this."
    ),
}

_GENERIC_APPROVED = [
    "Maintain a low debt-to-income ratio by keeping EMI payments under 40% of monthly income.",
    "Continue timely repayments on all existing loans to keep your credit score high.",
    "Building additional savings or investment assets will help you qualify for even better terms in future.",
]

_GENERIC_REJECTED = [
    "Work on building 6-12 months of clean repayment history before reapplying.",
    "Consider reducing your loan amount or extending the tenure to lower monthly EMI.",
    "A creditworthy co-applicant or guarantor can significantly improve approval chances.",
]


def generate_suggestions(top_features: list, decision: str) -> list:
    """
    Returns 3-4 actionable improvement suggestions based on negative factors.
    Falls back to generic advice if no specific templates match.
    """
    negative_features = [f for f in top_features if f["impact"] < 0]
    tips = []
    for f in negative_features:
        raw = f.get("raw_feature", "")
        tip = _SUGGESTIONS.get(raw)
        if tip:
            tips.append(tip)
        if len(tips) >= 4:
            break

    if not tips:
        return _GENERIC_REJECTED if decision != "Approved" else _GENERIC_APPROVED

    # Always append one general tip
    if decision == "Approved":
        tips.append(_GENERIC_APPROVED[0])
    else:
        tips.append(_GENERIC_REJECTED[2])

    return tips[:4]
