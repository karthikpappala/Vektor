
# ⚡ Vektor — AI-Powered Data Intelligence

> Upload any dataset. Instantly understand it. Know exactly what to do with it.

Vektor is an end-to-end data analysis assistant that takes any CSV or Excel file and automatically runs EDA, detects ML problem type, trains baseline models, generates an AI narrative, and lets you chat with your data — all in one place.

---

## Features

| Page | What it does |
|------|-------------|
| 🏠 **Home** | Upload dataset, instant quality score + preview |
| 📊 **EDA** | Column intelligence, distributions, correlations, quality alerts |
| 🎯 **ML Readiness** | Problem type detection, readiness score, preprocessing checklist |
| 🌲 **Baseline Models** | Auto-train RF + LightGBM + Logistic Regression with leakage detection & CV |
| 🧠 **AI Narrative** | Groq-powered plain-English story grounded in real computed facts |
| 💬 **Chat** | Ask anything about your dataset in plain English |
| 📄 **Preprocessing Script** | Download a ready-to-run Python script with your column names baked in |

---

## Tech Stack

- **Frontend:** Streamlit multi-page app
- **ML:** scikit-learn, LightGBM, pandas, numpy
- **AI:** Groq API (Llama 3.1-8b-instant)
- **Deployment:** Docker on HuggingFace Spaces

---

## How to Use

1. Upload a CSV or Excel file on the Home page
2. Navigate through the sidebar pages
3. For AI Narrative and Chat — enter your free [Groq API key](https://console.groq.com/keys)

---

## Local Setup

```bash
git clone https://github.com/karthikpappala/Vektor
cd Vektor
pip install -r requirements.txt
streamlit run app.py
```

---

Built by [Karthik Pappala](https://github.com/karthikpappala) · IIITDM Kurnool · AI & Data Science
