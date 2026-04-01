import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_model():
    with open("models/xgb_churn.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

def explain_customer(customer_data: pd.DataFrame, model=None, feature_cols=None):
    """
    Returns SHAP values for a single customer.
    customer_data: DataFrame row with feature columns
    """
    if model is None:
        model, feature_cols = load_model()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(customer_data[feature_cols])

    explanation = []
    for i, col in enumerate(feature_cols):
        explanation.append({
            "feature": col,
            "value": float(customer_data[col].values[0]),
            "shap_value": float(shap_values[0][i]),
            "impact": "increases churn risk" if shap_values[0][i] > 0 else "decreases churn risk"
        })

    explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return explanation[:5]  # Top 5 factors

def plot_summary(X: pd.DataFrame, model=None, feature_cols=None, save_path="data/shap_summary.png"):
    if model is None:
        model, feature_cols = load_model()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[feature_cols])

    plt.figure()
    shap.summary_plot(shap_values, X[feature_cols], show=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")

if __name__ == "__main__":
    from ml.train import load_and_preprocess
    model, feature_cols = load_model()
    X, y, ids, _ = load_and_preprocess()
    plot_summary(X, model, feature_cols)
