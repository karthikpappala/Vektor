"""
Vektor — AI-Powered Data Intelligence
Entry point: upload dataset, store in session state, navigate to pages.
"""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from modules.utils import load_dataframe, compute_quality_score, get_analytical_num_cols, is_id_column
from modules.eda import get_quality_alerts, get_strong_correlations
from modules.ml_readiness import (detect_problem_type, compute_ml_readiness,
                                   get_feature_importance, get_preprocessing_steps)

st.set_page_config(page_title="Vektor", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# ── Load shared CSS ───────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-eyebrow">AI-Powered Data Intelligence</div>
    <p class="hero-logo">Vektor</p>
    <p class="hero-tagline">Upload any dataset. Instantly understand it.<br>Know exactly what to do with it.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"],
                             label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 3rem 0;color:#2a2a4a;font-size:0.8rem;letter-spacing:1px;">
        Supports CSV · XLSX · XLS &nbsp;·&nbsp; Up to 200 MB
    </div>""", unsafe_allow_html=True)

    # Show feature cards when no file is uploaded
    c1, c2, c3 = st.columns(3)
    features = [
        ("📊", "EDA Report",       "Column intelligence, distributions, correlations, quality score"),
        ("🎯", "ML Readiness",     "Problem type detection, readiness score, preprocessing checklist"),
        ("🌲", "Baseline Models",  "Auto-train Random Forest, LightGBM & Logistic Regression with CV"),
        ("🧠", "AI Narrative",     "Groq-powered plain-English story of your data — grounded in facts"),
        ("💬", "Chat Interface",   "Ask anything about your dataset in plain English"),
        ("🛡️", "Leakage Guard",   "Automatic detection and exclusion of target-leaking columns"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(f"""<div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load & cache dataset ──────────────────────────────────────────────────────
@st.cache_data
def process_upload(file):
    df = load_dataframe(file)
    n_rows, n_cols = df.shape
    n_nulls  = int(df.isnull().sum().sum())
    n_dupes  = int(df.duplicated().sum())
    qs       = compute_quality_score(df)

    anal_num = get_analytical_num_cols(df)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    id_cols  = [c for c in df.select_dtypes(include="number").columns if is_id_column(df[c], c)]
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]

    alerts      = get_quality_alerts(df, anal_num, cat_cols, const_cols, id_cols)
    strong_corr = get_strong_correlations(df, anal_num) if len(anal_num) >= 2 else []

    problem_type, problem_reason, problem_icon, _ = detect_problem_type(df, cat_cols, anal_num, id_cols)
    ml_score, score_breakdown = compute_ml_readiness(df, anal_num, cat_cols, const_cols, id_cols)
    ml_verdict = ("Ready for training" if ml_score >= 75
                  else "Needs preprocessing" if ml_score >= 50
                  else "Significant cleanup needed")
    feat_importance = get_feature_importance(df, anal_num, cat_cols)
    prep_steps = get_preprocessing_steps(df, anal_num, cat_cols, const_cols, id_cols, problem_type)

    return {
        "df": df, "n_rows": n_rows, "n_cols": n_cols,
        "n_nulls": n_nulls, "n_dupes": n_dupes, "qs": qs,
        "anal_num": anal_num, "cat_cols": cat_cols,
        "id_cols": id_cols, "const_cols": const_cols,
        "high_null": alerts["high_null"], "high_skew": alerts["high_skew"],
        "strong_corr": strong_corr,
        "problem_type": problem_type, "problem_reason": problem_reason,
        "problem_icon": problem_icon,
        "ml_score": ml_score, "ml_verdict": ml_verdict,
        "score_breakdown": score_breakdown,
        "feat_importance": feat_importance, "prep_steps": prep_steps,
    }

try:
    data = process_upload(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# Store in session state so all pages can access it
st.session_state["vektor_data"] = data

# ── Quick summary after upload ────────────────────────────────────────────────
qs = data["qs"]
qs_color = "#5dbb6a" if qs >= 80 else "#d4a040" if qs >= 60 else "#e86060"

st.markdown(f"""
<div class="alert-ok" style="margin:1rem 0;">
    ✅ &nbsp;<strong>{uploaded.name}</strong> loaded — 
    <strong>{data['n_rows']:,} rows</strong>, 
    <strong>{data['n_cols']} columns</strong>, 
    quality score <strong style="color:{qs_color}">{qs}/100</strong>.
    Use the sidebar to explore each analysis.
</div>""", unsafe_allow_html=True)

# Preview
st.markdown('<p class="section-header">Data Preview</p>', unsafe_allow_html=True)
st.dataframe(data["df"].head(8), use_container_width=True)
