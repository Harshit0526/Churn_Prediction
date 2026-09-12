from __future__ import annotations

from typing import Any

from .config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


def risk_level(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def risk_color(level: str) -> str:
    return {"High": "red", "Medium": "orange", "Low": "green"}.get(level, "gray")


def recommend_retention(customer: dict[str, Any], level: str) -> list[dict[str, str]]:
    """Transparent rule-based recommendations for a customer's risk drivers."""
    recommendations: list[dict[str, str]] = []
    contract = customer.get("Contract", "Month-to-month")
    tenure = float(customer.get("tenure", 0))
    payment = customer.get("PaymentMethod", "")
    monthly_charges = float(customer.get("MonthlyCharges", 0))
    internet = customer.get("InternetService", "No")
    support = customer.get("TechSupport", "No")

    if level == "High":
        recommendations.append(
            {
                "priority": "Immediate",
                "action": "Assign a retention specialist within 48 hours",
                "reason": "High predicted churn risk needs a human-led save conversation.",
            }
        )
        if contract == "Month-to-month":
            recommendations.append(
                {
                    "priority": "High",
                    "action": "Offer a time-bound 12-month contract incentive",
                    "reason": "Month-to-month customers have less commitment friction.",
                }
            )
        if monthly_charges >= 80:
            recommendations.append(
                {
                    "priority": "High",
                    "action": "Review the bill and offer a targeted value bundle",
                    "reason": "High monthly charges can amplify perceived value risk.",
                }
            )
    elif level == "Medium":
        recommendations.append(
            {
                "priority": "Next best action",
                "action": "Send a personalized service health check-in",
                "reason": "Medium risk is an opportunity for proactive engagement.",
            }
        )
        if tenure <= 12:
            recommendations.append(
                {
                    "priority": "Next best action",
                    "action": "Trigger a new-customer onboarding journey",
                    "reason": "Early-tenure customers benefit from reassurance and education.",
                }
            )
    else:
        recommendations.append(
            {
                "priority": "Nurture",
                "action": "Enroll in loyalty and referral communications",
                "reason": "Low-risk customers are better served by relationship-building.",
            }
        )

    if payment in {"Electronic check", "Mailed check"}:
        recommendations.append(
            {
                "priority": "Preventive",
                "action": "Offer autopay setup assistance",
                "reason": "Manual payment methods create avoidable payment friction.",
            }
        )
    if internet != "No" and support == "No":
        recommendations.append(
            {
                "priority": "Preventive",
                "action": "Offer a complimentary technical support orientation",
                "reason": "Internet customers without support may need help before frustration compounds.",
            }
        )
    return recommendations
