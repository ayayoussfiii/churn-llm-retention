import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model():
    with open("models/xgb_churn.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

def preprocess_input(raw: dict) -> pd.DataFrame:
    """Convert raw customer dict to model-ready DataFrame."""
    contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    payment_map = {"Bank Transfer": 0, "Credit Card": 1, "Electronic Check": 2, "Mailed Check": 3}
    internet_map = {"DSL": 0, "Fiber": 1, "No": 2}

    processed = {
        "tenure_months": raw.get("tenure_months", 12),
        "monthly_charges": raw.get("monthly_charges", 50),
        "total_charges": raw.get("total_charges", 600),
        "num_products": raw.get("num_products", 2),
        "support_calls": raw.get("support_calls", 1),
        "late_payments": raw.get("late_payments", 0),
        "contract_type": contract_map.get(raw.get("contract_type", "Month-to-Month"), 0),
        "payment_method": payment_map.get(raw.get("payment_method", "Credit Card"), 1),
        "internet_service": internet_map.get(raw.get("internet_service", "Fiber"), 1),
        "online_security": raw.get("online_security", 0),
        "tech_support": raw.get("tech_support", 0),
        "satisfaction_score": raw.get("satisfaction_score", 3),
    }
    return pd.DataFrame([processed])

def predict(raw: dict):
    model, feature_cols = load_model()
    df = preprocess_input(raw)
    prob = model.predict_proba(df[feature_cols])[0][1]
    label = "CHURN" if prob > 0.5 else "RETAIN"
    return {
        "churn_probability": round(float(prob), 4),
        "prediction": label,
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
        "dataframe": df
    }

if __name__ == "__main__":
    sample = {
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
    result = predict(sample)
    print(f"Prediction: {result['prediction']}")
    print(f"Churn Probability: {result['churn_probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
