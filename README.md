# ChurnAI — Customer Churn Prediction +  LLM Retention Strategy

> End-to-end AI system that predicts customer churn with XGBoost, explains predictions with SHAP, and automatically generates personalized retention strategies using Claude (Anthropic).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-AA0000?style=flat)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-00A67E?style=flat)
![Claude](https://img.shields.io/badge/LLM-Claude%20API-D4A27A?style=flat)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.91-success?style=flat)

---

## Overview

ChurnAI helps telecom companies identify customers at risk of churning **before** they leave. It combines:

- A trained **XGBoost classifier** for churn probability scoring
- **SHAP explainability** to surface the top risk factors per customer
- A **Claude LLM** that reads those risk factors and generates a personalized 90-day retention plan
- A **Streamlit dashboard** for real-time interactive analysis

---

## Pipeline

```
Customer Data → XGBoost Model → SHAP Explainability → Claude LLM → Streamlit Dashboard
(CSV / Form)    (Churn score)    (Top risk factors)   (90-day plan)  (Interactive UI)
```

---

## Stack

| Layer | Technology |
|---|---|
| ML model | `XGBoost 2.0+` |
| Explainability | `SHAP` |
| LLM | `Anthropic Claude` via API |
| Dashboard | `Streamlit` + `Plotly` |
| Language | `Python 3.10+` |
| Data | IBM Telco Dataset (7,043 customers) |

---

## Model Performance

Evaluated on a held-out test set (20% split), trained on 12 features including tenure, contract type, monthly charges, internet service, and payment method.

| Metric | Score |
|---|---|
| ROC-AUC | ~0.91 |
| Precision (Churn) | ~0.84 |
| Recall (Churn) | ~0.79 |
| F1-Score (Churn) | ~0.81 |
| Accuracy | ~0.83 |

---

## Key Features

- **ML prediction** — XGBoost trained on 12 customer features with probability calibration
- **SHAP explainability** — top risk drivers per individual customer, visualized as bar charts
- **LLM retention plans** — Claude generates personalized 90-day retention action plans
- **Interactive dashboard** — real-time risk gauge, SHAP bar charts, dataset overview
- **Risk segmentation** — customers classified as Low / Medium / High churn risk
- **Dataset analytics** — churn rate by contract type, tenure group, and payment method

---

## Project Structure

```
churn-llm-retention/
├── data/
│   ├── generate_dataset.py     # Synthetic churn dataset generator
│   └── churn_dataset.csv       # Generated dataset (1000 customers)
├── ml/
│   ├── train.py                # XGBoost model training + evaluation
│   ├── explain.py              # SHAP explainability helpers
│   └── predict.py              # Inference pipeline
├── llm/
│   └── retention.py            # Claude LLM strategy generator
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── models/                     # Saved model artifacts (.pkl)
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/ayayoussfiii/churn-llm-retention.git
cd churn-llm-retention

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file at the root of the project:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

---

## Quickstart

```bash
# 1. Generate synthetic dataset
python data/generate_dataset.py

# 2. Train XGBoost model
python ml/train.py

# 3. Launch dashboard
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Roadmap

- [ ] Batch CSV prediction — upload multiple customers at once
- [ ] Export retention strategies as PDF
- [ ] Model retraining pipeline
- [ ] REST API with FastAPI
- [ ] Docker support

---

Built with Python · Powered by XGBoost 
