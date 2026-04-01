import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

def load_and_preprocess(path="data/churn_dataset.csv"):
    df = pd.read_csv(path)

    # Drop non-feature columns
    df = df.drop(columns=["customerid"], errors="ignore")

    # Fix totalcharges
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)

    # Encode all object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col != "churn":
            df[col] = le.fit_transform(df[col].astype(str))

    # Encode target
    if df["churn"].dtype == object:
        df["churn"] = (df["churn"] == "Yes").astype(int)

    feature_cols = [c for c in df.columns if c != "churn"]
    X = df[feature_cols]
    y = df["churn"]
    ids = pd.Series(range(len(df)))
    return X, y, ids, feature_cols

def train():
    X, y, ids, feature_cols = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Model Performance ===")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs("models", exist_ok=True)
    with open("models/xgb_churn.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print("\nModel saved to models/xgb_churn.pkl")
    return model, feature_cols

if __name__ == "__main__":
    train()