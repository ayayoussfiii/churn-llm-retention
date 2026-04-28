#  ChurnAI — Customer Churn Prediction + LLM Retention Strategy

> An end-to-end AI system that predicts customer churn with XGBoost, explains predictions with SHAP, and automatically generates personalized retention strategies using Claude (Anthropic).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Anthropic](https://img.shields.io/badge/Claude-Anthropic-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Overview

ChurnAI helps telecom companies identify customers at risk of churning **before** they leave. It combines:

- A trained **XGBoost classifier** for churn probability scoring
- **SHAP explainability** to surface the top risk factors per customer
- A **Claude LLM** that reads those risk factors and generates a personalized 90-day retention plan
- A **Streamlit dashboard** for real-time interactive analysis

---

##  Architecture

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

---

## 🛠️ Stack

| Layer | Technology |
|---|---|
| ML Model | `XGBoost 2.0+` |
| Explainability | `SHAP` |
| LLM | `Anthropic Claude` via API |
| Dashboard | `Streamlit` + `Plotly` |
| Language | `Python 3.10+` |
| Data | IBM Telco Dataset (7,043 customers) |

---

##  Project Structure

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

##  Setup

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/ayayoussfiii/churn-llm-retention.git
cd churn-llm-retention

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the root:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

---

##  Run

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

##  Model Performance

Evaluated on a held-out test set (20% split):

| Metric | Score |
|---|---|
| ROC-AUC | ~0.91 |
| Precision (Churn) | ~0.84 |
| Recall (Churn) | ~0.79 |
| F1-Score (Churn) | ~0.81 |
| Accuracy | ~0.83 |

> Model trained on 12 features including tenure, contract type, monthly charges, internet service, and payment method.

---

##  Features

- **ML Prediction** — XGBoost trained on 12 customer features with probability calibration
- **Explainability** — SHAP values highlight the top risk drivers per individual customer
- **LLM Retention** — Claude generates personalized 90-day retention action plans
- **Interactive Dashboard** — Real-time risk gauge, SHAP bar charts, dataset overview
- **Risk Segmentation** — Customers classified as Low / Medium / High churn risk
- **Dataset Analytics** — Churn rate by contract type, tenure group, and payment method

---

## 🗺️ Roadmap

- [ ] Add batch CSV prediction (upload multiple customers at once)
- [ ] Export retention strategy as PDF
- [ ] Add model retraining pipeline
- [ ] REST API with FastAPI
- [ ] Docker support

---


---

📍 Built with Python · Powered by XGBoost + Claude (Anthropic)
