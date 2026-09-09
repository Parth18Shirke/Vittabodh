"""
Explainable Loan Risk Scoring (ELRS) engine.

Implements the five risk-pillar formulas from the VittaBodh paper exactly:
  ELRS = CH + IS + AC + DB + PR   (0-100)

  CH  (Credit Health,        25%): ((CreditScore-300)/600)*18 - 2.5*Defaults - 0.5*LatePayments
  IS  (Income & Stability,   25%): 12*(DSCR/3) + 7*(YearsExperience/20) + 6*(JobDuration/60)
  AC  (Asset & Collateral,   20%): 16*(AssetRatio/2) + 4*Guarantor
  DB  (Debt Burden,          15%): 15 - 2.5*ExistingLoans - 10*(ExistingEMI/MonthlyIncome)
  PR  (Profile Risk,         15%): 5*EducationScore + 6*EmploymentScore + 4*AgeScore
"""
from dataclasses import dataclass

EDUCATION_SCORE = {
    "PhD": 1.0, "Master": 0.85, "Bachelor": 0.7, "Diploma": 0.5, "High School": 0.35,
}
EMPLOYMENT_SCORE = {
    "Government": 1.0, "Salaried": 0.85, "Self-Employed": 0.6, "Business": 0.55, "Contract": 0.4,
}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class ELRSResult:
    credit_health: float
    income_stability: float
    asset_collateral: float
    debt_burden: float
    profile_risk: float
    total: float
    band: str
    dscr: float
    asset_ratio: float

    def as_dict(self):
        return {
            "credit_health": round(self.credit_health, 2),
            "income_stability": round(self.income_stability, 2),
            "asset_collateral": round(self.asset_collateral, 2),
            "debt_burden": round(self.debt_burden, 2),
            "profile_risk": round(self.profile_risk, 2),
            "total": round(self.total, 2),
            "band": self.band,
            "dscr": round(self.dscr, 2),
            "asset_ratio": round(self.asset_ratio, 2),
        }


def _risk_band(score: float) -> str:
    if score >= 75:
        return "Low Risk"
    if score >= 55:
        return "Moderate Risk"
    if score >= 35:
        return "Elevated Risk"
    return "High Risk"


def estimate_emi(loan_amount: float, tenure_months: int, annual_rate: float = 0.11) -> float:
    """Estimated EMI for the requested loan, used inside the DSCR term."""
    if tenure_months <= 0:
        return loan_amount
    r = annual_rate / 12
    if r == 0:
        return loan_amount / tenure_months
    factor = (1 + r) ** tenure_months
    return loan_amount * r * factor / (factor - 1)


def compute_elrs(applicant: dict) -> ELRSResult:
    credit_score = float(applicant["CreditScore"])
    defaults = float(applicant["PastDefaults"])
    late_payments = float(applicant["LatePayments"])

    monthly_income = max(float(applicant["MonthlyIncome"]), 1.0)
    monthly_expenses = float(applicant["MonthlyExpenses"])
    existing_emi = float(applicant["ExistingEMI"])
    years_experience = float(applicant["YearsExperience"])
    job_duration = float(applicant["CurrentJobDuration"])

    savings = float(applicant["SavingsBalance"])
    investment = float(applicant["InvestmentValue"])
    collateral = float(applicant["CollateralValue"])
    loan_amount = max(float(applicant["LoanAmount"]), 1.0)
    guarantor = 1.0 if str(applicant["GuarantorAvailable"]).strip().lower() == "yes" else 0.0

    existing_loans = float(applicant["ExistingLoans"])

    education = applicant["Education"]
    employment = applicant["EmploymentType"]
    age = float(applicant["Age"])

    tenure = int(applicant["LoanTenure"])
    est_emi = estimate_emi(loan_amount, tenure)

    # 1) Credit Health
    ch = ((credit_score - 300) / 600) * 18 - 2.5 * defaults - 0.5 * late_payments
    ch = _clip(ch, 0, 25)

    # 2) Income & Stability
    dscr = (monthly_income - monthly_expenses - existing_emi) / max(est_emi, 1.0)
    is_score = 12 * (dscr / 3) + 7 * (years_experience / 20) + 6 * (job_duration / 60)
    is_score = _clip(is_score, 0, 25)

    # 3) Asset & Collateral
    asset_ratio = (savings + investment + collateral) / loan_amount
    ac = 16 * (asset_ratio / 2) + 4 * guarantor
    ac = _clip(ac, 0, 20)

    # 4) Debt Burden
    db = 15 - 2.5 * existing_loans - 10 * (existing_emi / monthly_income)
    db = _clip(db, 0, 15)

    # 5) Profile Risk
    edu_score = EDUCATION_SCORE.get(education, 0.5)
    emp_score = EMPLOYMENT_SCORE.get(employment, 0.5)
    age_score = 1 - abs(age - 38) / 38
    pr = 5 * edu_score + 6 * emp_score + 4 * age_score
    pr = _clip(pr, 0, 15)

    total = ch + is_score + ac + db + pr
    total = _clip(total, 0, 100)

    return ELRSResult(
        credit_health=ch,
        income_stability=is_score,
        asset_collateral=ac,
        debt_burden=db,
        profile_risk=pr,
        total=total,
        band=_risk_band(total),
        dscr=dscr,
        asset_ratio=asset_ratio,
    )
