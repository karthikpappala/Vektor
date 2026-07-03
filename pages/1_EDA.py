"""Page 1 — Exploratory Data Analysis"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import matplotlib.pyplot as plt

from modules.utils import detect_dtype_label, null_class, skewness_label, get_outlier_count
from modules.eda import plot_distributions, plot_correlation, get_strong_correlations

st.set_page_config(page_title="Vektor · EDA", page_icon="📊", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "vektor_data" not in st.session_state:
    st.warning("⚠ Please upload a dataset on the Home page first.")
    st.stop()

d = st.session_state["vektor_data"]
df          = d["df"]
anal_num    = d["anal_num"]
cat_cols    = d["cat_cols"]
id_cols     = d["id_cols"]
const_cols  = d["const_cols"]
high_null   = d["high_null"]
high_skew   = d["high_skew"]
n_rows      = d["n_rows"]
n_cols      = d["n_cols"]
n_nulls     = d["n_nulls"]
n_dupes     = d["n_dupes"]
qs          = d["qs"]

qs_color = "#5dbb6a" if qs >= 80 else "#d4a040" if qs >= 60 else "#e86060"

# ── Overview ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
for col_w, label, value, sub in zip(
    [c1,c2,c3,c4,c5],
    ["Rows","Columns","Missing Values","Duplicates","Quality Score"],
    [f"{n_rows:,}", str(n_cols), f"{n_nulls:,}", str(n_dupes), str(qs)],
    ["observations","features",f"{df.isnull().mean().mean()*100:.1f}% of cells",
     "exact matches","out of 100"]
):
    color = qs_color if label == "Quality Score" else "#e2e2f0"
    with col_w:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ── Column Intelligence ───────────────────────────────────────────────────────
st.markdown('<p class="section-header">Column Intelligence</p>', unsafe_allow_html=True)
rows_html = ""
for col in df.columns:
    dtype_label, badge_cls = detect_dtype_label(df[col], col)
    null_pct = df[col].isnull().mean() * 100
    n_unique = df[col].nunique()
    is_const = n_unique <= 1

    if dtype_label == "numeric":
        skew = df[col].skew()
        skew_txt, skew_col = skewness_label(skew)
        out = get_outlier_count(df[col].dropna())
        extra = f'<span style="color:{skew_col};font-size:0.73rem">{skew_txt}</span>&nbsp;&nbsp;<span style="color:#3a3a5a;font-size:0.72rem">{out} outliers</span>'
    elif dtype_label == "id / key":
        extra = '<span style="color:#3a6a7a;font-size:0.72rem">identifier — excluded</span>'
    elif dtype_label == "categorical" and n_unique > 0:
        top = str(df[col].value_counts().index[0])[:22]
        extra = f'<span style="color:#3a3a5a;font-size:0.72rem">top: <span style="color:#9090b0">{top}</span></span>'
    else:
        extra = ""

    row_style = ' style="opacity:0.45;"' if is_const else ''
    const_tag = "&nbsp;<span style='color:#3a3a5a;font-size:0.65rem'>[const]</span>" if is_const else ""
    rows_html += (f'<tr{row_style}><td class="col-name">{col}{const_tag}</td>'
                  f'<td><span class="badge {badge_cls}">{dtype_label}</span></td>'
                  f'<td class="{null_class(null_pct)}">{null_pct:.1f}%</td>'
                  f'<td>{n_unique:,}</td><td>{extra}</td></tr>')

st.markdown(f"""<table class="col-table"><thead><tr>
    <th>Column</th><th>Type</th><th>Nulls</th><th>Unique</th><th>Quick Insight</th>
</tr></thead><tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

# ── Alerts ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Data Quality Alerts</p>', unsafe_allow_html=True)
alerts = []
if high_null:  alerts.append(f"High nulls in: <strong>{', '.join(high_null)}</strong>")
if n_dupes:    alerts.append(f"<strong>{n_dupes} duplicate rows</strong> detected")
if high_skew:  alerts.append(f"High skewness in: <strong>{', '.join(high_skew)}</strong> — log transform recommended")
if const_cols: alerts.append(f"Constant columns (drop them): <strong>{', '.join(const_cols)}</strong>")
if id_cols:
    st.markdown(f'<div class="alert-info">ℹ &nbsp;<strong>{", ".join(id_cols)}</strong> detected as ID column(s) — excluded from plots.</div>', unsafe_allow_html=True)
if not alerts:
    st.markdown('<div class="alert-ok">✓ No major data quality issues detected.</div>', unsafe_allow_html=True)
else:
    for msg in alerts:
        st.markdown(f'<div class="alert-warn">⚠ &nbsp;{msg}</div>', unsafe_allow_html=True)

# ── Distributions ─────────────────────────────────────────────────────────────
if anal_num:
    st.markdown('<p class="section-header">Numeric Distributions</p>', unsafe_allow_html=True)
    fig = plot_distributions(df, anal_num)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Correlation ───────────────────────────────────────────────────────────────
if len(anal_num) >= 2:
    st.markdown('<p class="section-header">Correlation Matrix</p>', unsafe_allow_html=True)
    fig2 = plot_correlation(df, anal_num)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    strong = get_strong_correlations(df, anal_num)
    if strong:
        st.markdown('<p class="section-header">Strong Correlations (> 0.8)</p>', unsafe_allow_html=True)
        for a, b, v in strong:
            st.markdown(f'<div class="alert-info">⚡ <strong>{a}</strong> ↔ <strong>{b}</strong> · r = {v}</div>', unsafe_allow_html=True)

# ── Sample ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Sample Data (first 10 rows)</p>', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)
