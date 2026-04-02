# 🧠 ChurnAI — Customer Churn Prediction + LLM Retention Strategy

> An end-to-end AI system that predicts customer churn with XGBoost, explains predictions with SHAP, and automatically generates personalized retention strategies .

---

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Customer Data  │───▶│  XGBoost Model   │───▶│  Churn Probability  │
│  (CSV / Form)   │    │  (trained ML)    │    │  + Risk Level       │
└─────────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                           │
                                               ┌───────────▼───────────┐
                                               │   SHAP Explainability │
                                               │   (Top risk factors)  │
                                               └───────────┬───────────┘
                                                           │
                                               ┌───────────▼───────────┐
                                               │  Claude LLM (Anthropic)│
                                               │  Retention Strategy    │
                                               │  Generator             │
                                               └───────────┬───────────┘
                                                           │
                                               ┌───────────▼───────────┐
                                               │  Streamlit Dashboard  │
                                               │  (Interactive UI)     │
                                               └───────────────────────┘
```

## Stack

`Python 3.10+` `XGBoost` `SHAP` `Anthropic Claude` `Streamlit` `Plotly`

## Project Structure

```
churn-llm-retention/
├── data/
│   ├── generate_dataset.py     # Synthetic churn dataset generator
│   └── churn_dataset.csv       # Generated dataset (1000 customers)
├── ml/
│   ├── train.py                # XGBoost model training
│   ├── explain.py              # SHAP explainability
│   └── predict.py              # Inference pipeline
├── llm/
│   └── retention.py            # Claude LLM strategy generator
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── models/                     # Saved model artifacts
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/churn-llm-retention.git
cd churn-llm-retention

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here  # Windows: set ANTHROPIC_API_KEY=your_key
```

## Run

```bash
# 1. Generate synthetic dataset
python data/generate_dataset.py

# 2. Train XGBoost model
python ml/train.py

# 3. Launch dashboard
streamlit run dashboard/app.py
```

## Results

| Metric | Score |
|---|---|
| ROC-AUC | ~0.91 |
| Precision (Churn) | ~0.84 |
| Recall (Churn) | ~0.79 |
| F1-Score | ~0.81 |

## Features

- **ML Prediction** — XGBoost trained on 12 customer features
- **Explainability** — SHAP values highlight top churn drivers per customer
- **LLM Retention** —  generates personalized 90-day retention plans
- **Interactive Dashboard** — Real-time risk gauge, SHAP charts, dataset overview

---

📍 Built with Python 
