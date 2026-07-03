"""Page 2 — ML Readiness"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(page_title="Vektor · ML Readiness", page_icon="🎯", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "vektor_data" not in st.session_state:
    st.warning("⚠ Please upload a dataset on the Home page first.")
    st.stop()

d = st.session_state["vektor_data"]

ml_score       = d["ml_score"]
ml_verdict     = d["ml_verdict"]
score_breakdown= d["score_breakdown"]
problem_type   = d["problem_type"]
problem_reason = d["problem_reason"]
problem_icon   = d["problem_icon"]
feat_importance= d["feat_importance"]
prep_steps     = d["prep_steps"]

qs_color = "#5dbb6a" if ml_score >= 75 else "#d4a040" if ml_score >= 50 else "#e86060"

# ── Problem type ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Problem Type Detection</p>', unsafe_allow_html=True)
col_left, col_right = st.columns([1.4, 1])
with col_left:
    st.markdown(f"""<div class="problem-card">
        <div class="problem-type-label">Detected Problem Type</div>
        <div class="problem-type-value">{problem_icon} &nbsp;{problem_type}</div>
        <div class="problem-type-reason">{problem_reason}</div>
    </div>""", unsafe_allow_html=True)
with col_right:
    st.markdown(f"""<div class="score-card">
        <div class="score-sublabel">ML Readiness Score</div>
        <div class="score-number" style="color:{qs_color}">{ml_score}</div>
        <div class="score-verdict" style="color:{qs_color}">{ml_verdict}</div>
    </div>""", unsafe_allow_html=True)

# ── Readiness breakdown ───────────────────────────────────────────────────────
st.markdown('<p class="section-header">Readiness Breakdown</p>', unsafe_allow_html=True)
dim_colors = {"Completeness":"#7c5cf8","Size Adequacy":"#a855f7",
              "Feature Quality":"#5dbb6a","Numeric Health":"#d4a040","Label Readiness":"#4ab8d4"}
bd_html = ""
for dim, val in score_breakdown.items():
    color = dim_colors.get(dim, "#7c5cf8")
    bd_html += (f'<div class="readiness-row">'
                f'<div class="readiness-label">{dim}</div>'
                f'<div class="readiness-bar-wrap"><div class="readiness-bar-fill" '
                f'style="width:{val}%;background:{color};opacity:0.85;"></div></div>'
                f'<div class="readiness-score-val">{val}</div></div>')
st.markdown(f'<div style="background:#0a0a18;border:1px solid #14143a;border-radius:12px;padding:1.5rem 1.8rem;">{bd_html}</div>', unsafe_allow_html=True)

# ── Feature signal ────────────────────────────────────────────────────────────
if feat_importance:
    st.markdown('<p class="section-header">Feature Signal Strength (heuristic)</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#3a3a5a;font-size:0.75rem;margin:-0.5rem 0 1rem 0;">Based on variance and entropy — not model-trained.</p>', unsafe_allow_html=True)
    fi_html = ""
    for i, (col, val) in enumerate(feat_importance):
        opacity = max(1.0 - i * 0.05, 0.4)
        fi_html += (f'<div class="feat-row"><div class="feat-name" title="{col}">{col}</div>'
                    f'<div class="feat-bar-wrap"><div class="feat-bar-fill" style="width:{val}%;opacity:{opacity};"></div></div>'
                    f'<div class="feat-val">{val}</div></div>')
    st.markdown(f'<div style="background:#0a0a18;border:1px solid #14143a;border-radius:12px;padding:1.5rem 1.8rem;">{fi_html}</div>', unsafe_allow_html=True)

# ── Preprocessing checklist ───────────────────────────────────────────────────
st.markdown('<p class="section-header">Preprocessing Checklist</p>', unsafe_allow_html=True)
for priority, icon, title, desc in prep_steps:
    st.markdown(f"""<div class="prep-item prep-priority-{priority}">
        <div class="prep-icon">{icon}</div>
        <div><div class="prep-title">{title}</div>
        <div class="prep-desc">{desc}</div></div>
    </div>""", unsafe_allow_html=True)
