import anthropic
import json

client = anthropic.Anthropic()

def generate_retention_strategy(customer_data: dict, churn_probability: float, shap_factors: list) -> str:
    """
    Uses Claude to generate a personalized retention strategy.
    
    Args:
        customer_data: Raw customer attributes
        churn_probability: Model's predicted churn probability
        shap_factors: Top SHAP factors explaining the prediction
    """

    factors_text = "\n".join([
        f"  - {f['feature'].replace('_', ' ').title()}: {f['value']} → {f['impact']} (SHAP: {f['shap_value']:+.3f})"
        for f in shap_factors
    ])

    prompt = f"""You are a senior Customer Retention Strategist at a telecom company.

A customer has been flagged by our AI churn prediction model with the following profile:

## Customer Profile
- Tenure: {customer_data.get('tenure_months', 'N/A')} months
- Monthly Charges: ${customer_data.get('monthly_charges', 'N/A')}
- Contract Type: {customer_data.get('contract_type', 'N/A')}
- Support Calls (last 6 months): {customer_data.get('support_calls', 'N/A')}
- Late Payments: {customer_data.get('late_payments', 'N/A')}
- Satisfaction Score: {customer_data.get('satisfaction_score', 'N/A')}/5
- Internet Service: {customer_data.get('internet_service', 'N/A')}
- Online Security: {'Yes' if customer_data.get('online_security') else 'No'}
- Tech Support: {'Yes' if customer_data.get('tech_support') else 'No'}

## AI Model Output
- **Churn Probability: {churn_probability:.1%}**
- Risk Level: {"High 🔴" if churn_probability > 0.7 else "Medium 🟡" if churn_probability > 0.4 else "Low 🟢"}

## Top Churn Risk Factors (SHAP Explainability)
{factors_text}

## Your Task
Generate a **personalized retention strategy** with:

1. **Root Cause Analysis** — Why is this customer likely to churn? (2-3 sentences)
2. **Immediate Actions** (within 24h) — 3 specific actions a retention agent should take NOW
3. **Offer Recommendations** — 2-3 personalized offers/incentives tailored to this customer's profile
4. **Long-term Engagement Plan** — A 90-day plan to improve satisfaction and loyalty
5. **Risk Mitigation Score** — Estimate how much these actions could reduce churn probability

Be specific, actionable, and data-driven. Reference the actual customer data and SHAP factors in your recommendations.
"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def generate_batch_strategies(customers: list) -> list:
    """Generate retention strategies for multiple customers."""
    results = []
    for customer in customers:
        strategy = generate_retention_strategy(
            customer["data"],
            customer["churn_probability"],
            customer["shap_factors"]
        )
        results.append({
            "customer_id": customer.get("customer_id"),
            "churn_probability": customer["churn_probability"],
            "strategy": strategy
        })
    return results


if __name__ == "__main__":
    # Demo
    sample_customer = {
        "tenure_months": 3,
        "monthly_charges": 95,
        "total_charges": 285,
        "num_products": 1,
        "support_calls": 7,
        "late_payments": 3,
        "contract_type": "Month-to-Month",
        "payment_method": "Electronic Check",
        "internet_service": "Fiber",
        "online_security": 0,
        "tech_support": 0,
        "satisfaction_score": 2,
    }

    sample_shap = [
        {"feature": "support_calls", "value": 7, "shap_value": 0.45, "impact": "increases churn risk"},
        {"feature": "satisfaction_score", "value": 2, "shap_value": 0.38, "impact": "increases churn risk"},
        {"feature": "contract_type", "value": 0, "shap_value": 0.31, "impact": "increases churn risk"},
        {"feature": "tenure_months", "value": 3, "shap_value": 0.22, "impact": "increases churn risk"},
        {"feature": "late_payments", "value": 3, "shap_value": 0.18, "impact": "increases churn risk"},
    ]

    print("Generating retention strategy with Claude...")
    strategy = generate_retention_strategy(sample_customer, 0.87, sample_shap)
    print(strategy)
