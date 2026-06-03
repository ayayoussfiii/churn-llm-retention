<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.0+-AA0000?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SHAP-Explainability-00A67E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Claude-Anthropic-D4A27A?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ROC--AUC-0.91-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>

# ChurnAI

**End-to-end customer churn prediction with XGBoost, SHAP explainability,
and Claude-powered personalized retention strategies.**

[Features](#-features) · [Architecture](#-architecture) · [Performance](#-model-performance) · [Quickstart](#-quickstart) · [Roadmap](#-roadmap)

</div>

---

## What is ChurnAI?

ChurnAI helps telecom companies identify customers at risk of churning — **before they leave**.

Rather than just flagging who might churn, ChurnAI explains *why* and automatically generates a **personalized 90-day retention plan** per customer using Claude (Anthropic). All of this is surfaced through an interactive Streamlit dashboard, making it usable by both data scientists and business teams.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   CSV / Form │────▶│   XGBoost    │────▶│     SHAP     │────▶│  Claude LLM  │────▶│  Streamlit   │
│  (raw data)  │     │ (churn score)│     │ (risk factors│     │ (90-day plan)│     │  (dashboard) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## Features

| | Feature | Description |
|---|---|---|
| `ML` | **Churn prediction** | XGBoost trained on 12 customer features with probability calibration |
| `XAI` | **SHAP explainability** | Top risk drivers per customer — no black box, full transparency |
| `LLM` | **Retention plans** | Claude generates a tailored 90-day action plan per at-risk customer |
| `UI` | **Interactive dashboard** | Real-time risk gauge, SHAP bar charts, and full dataset analytics |
| `ML` | **Risk segmentation** | Customers auto-classified as Low / Medium / High churn risk |
| `Analytics` | **Dataset insights** | Churn breakdown by contract type, tenure group, and payment method |

---

## Model Performance

> Evaluated on a **20% held-out test set** · Trained on 12 features: tenure, contract type, monthly charges, internet service, payment method, and more.

| Metric | Score |
|---|---|
| ROC-AUC | **~0.91** |
| Precision (Churn) | ~0.84 |
| Recall (Churn) | ~0.79 |
| F1-Score (Churn) | ~0.81 |
| Accuracy | ~0.83 |

---

## Stack

| Layer | Technology |
|---|---|
| ML model | `XGBoost 2.0+` |
| Explainability | `SHAP` |
| LLM | `Claude API` (Anthropic) |
| Dashboard | `Streamlit` + `Plotly` |
| Language | `Python 3.10+` |
| Dataset | IBM Telco · 7,043 customers |

---

## Project Structure

```
churn-llm-retention/
├── data/
│   ├── generate_dataset.py     # Synthetic churn dataset generator
│   └── churn_dataset.csv       # Generated dataset (1 000 customers)
├── ml/
│   ├── train.py                # XGBoost training + evaluation
│   ├── explain.py              # SHAP explainability helpers
│   └── predict.py              # Inference pipeline
├── llm/
│   └── retention.py            # Claude retention strategy generator
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── models/                     # Saved model artifacts (.pkl)
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

---

## Quickstart

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

### Environment

```env
# .env
ANTHROPIC_API_KEY=your_api_key_here
```

### Run

```bash
# Generate synthetic dataset
python data/generate_dataset.py

# Train the XGBoost model
python ml/train.py

# Launch the dashboard → http://localhost:8501
streamlit run dashboard/app.py
```

---

## Roadmap

- [ ] **Batch CSV prediction** — analyze multiple customers in one upload
- [ ] **PDF export** — download retention strategy reports
- [ ] **Retraining pipeline** — automated model updates on new data
- [ ] **REST API** — FastAPI wrapper for production integration
- [ ] **Docker support** — one-command deployment

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

<div align="center">

Built with Python · Powered by **XGBoost** + **Claude** (Anthropic)

⭐ Star this repo if it was useful to you

</div>
