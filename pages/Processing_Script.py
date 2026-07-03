"""Page 6 — Auto-Generated Preprocessing Script"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from modules.script_generator import generate_preprocessing_script

st.set_page_config(page_title="Vektor · Script", page_icon="📄", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "vektor_data" not in st.session_state:
    st.warning("⚠ Please upload a dataset on the Home page first.")
    st.stop()

d  = st.session_state["vektor_data"]
df = d["df"]

st.markdown('<p class="section-header">Preprocessing Script</p>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#3a3a5a;font-size:0.78rem;margin:-0.5rem 0 1.5rem 0;">'
    'A ready-to-run Python script generated specifically for your dataset — '
    'real column names, real steps, no generic templates.</p>',
    unsafe_allow_html=True
)

# ── What's included summary ───────────────────────────────────────────────────
high_null  = d.get("high_null", [])
high_skew  = d.get("high_skew", [])
const_cols = d.get("const_cols", [])
id_cols    = d.get("id_cols", [])
anal_num   = d.get("anal_num", [])
cat_cols   = d.get("cat_cols", [])
low_card   = [c for c in cat_cols if 2 <= df[c].nunique() <= 15 and c not in high_null]
high_card  = [c for c in cat_cols if df[c].nunique() > 15 and c not in high_null]
num_impute = [c for c in anal_num if df[c].isnull().sum() > 0 and c not in high_null]

steps = [
    ("🗑️", "Drop columns",     f"{len(const_cols + id_cols)} constant/ID columns removed"),
    ("⚠️", "High-null drop",   f"{len(high_null)} column(s) with >30% missing" if high_null else "None detected"),
    ("🔧", "Imputation",        f"{len(num_impute)} numeric (median) · {len([c for c in cat_cols if df[c].isnull().sum()>0 and c not in high_null])} categorical (mode)"),
    ("📐", "Skew fix",          f"log1p on {len(high_skew)} column(s): {', '.join(high_skew[:3])}" if high_skew else "None detected"),
    ("🔤", "Encoding",          f"OHE on {len(low_card)} · Label on {len(high_card)} column(s)"),
    ("⚖️", "Scaling",           f"StandardScaler on {len(anal_num)} numeric column(s)"),
    ("✂️", "Train/test split",  "80/20 split with random_state=42"),
]

cols = st.columns(4)
for i, (icon, label, detail) in enumerate(steps):
    with cols[i % 4]:
        st.markdown(f"""<div class="metric-card" style="margin-bottom:0.6rem;">
            <div style="font-size:1.2rem;">{icon}</div>
            <div style="font-size:0.72rem;font-weight:700;color:#c8c8e8;margin:0.3rem 0 0.2rem 0;">{label}</div>
            <div style="font-size:0.68rem;color:#4a4a6a;line-height:1.4;">{detail}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div style="margin:1.5rem 0 0.5rem 0;"></div>', unsafe_allow_html=True)

# ── Generate script ───────────────────────────────────────────────────────────
if "preprocessing_script" not in st.session_state:
    script = generate_preprocessing_script(df, d)
    st.session_state["preprocessing_script"] = script

script = st.session_state["preprocessing_script"]

# ── Download button ───────────────────────────────────────────────────────────
st.download_button(
    label="⬇️  Download preprocessing.py",
    data=script,
    file_name="preprocessing.py",
    mime="text/x-python",
    use_container_width=True,
)

# ── Preview ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Script Preview</p>', unsafe_allow_html=True)
st.code(script, language="python", line_numbers=True)

# ── Usage instructions ────────────────────────────────────────────────────────
st.markdown('<p class="section-header">How to Use</p>', unsafe_allow_html=True)
st.markdown("""
<div class="alert-info">
    <strong>1.</strong> Download the script above<br>
    <strong>2.</strong> Place it in the same folder as your dataset<br>
    <strong>3.</strong> Edit line 1: replace <code>'your_dataset.csv'</code> with your actual filename<br>
    <strong>4.</strong> Edit the <code>TARGET</code> variable near the bottom with your target column name<br>
    <strong>5.</strong> Run: <code>python preprocessing.py</code>
</div>
""", unsafe_allow_html=True)
