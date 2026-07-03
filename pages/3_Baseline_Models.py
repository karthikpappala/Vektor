"""Page 3 — Baseline Model Training"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from modules.baseline_models import train_models

st.set_page_config(page_title="Vektor · Models", page_icon="🌲", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "vektor_data" not in st.session_state:
    st.warning("⚠ Please upload a dataset on the Home page first.")
    st.stop()

d       = st.session_state["vektor_data"]
df      = d["df"]
id_cols = d["id_cols"]

st.markdown('<p class="section-header">Baseline Model Training</p>', unsafe_allow_html=True)

all_candidates = [c for c in df.columns if c not in id_cols and df[c].nunique() > 1]
target_keywords = ["label","target","class","category","type","status","result","outcome","score","grade"]
suggested = [c for c in all_candidates if any(kw in c.lower() for kw in target_keywords)]
default_idx = all_candidates.index(suggested[0]) if suggested and suggested[0] in all_candidates else 0

target_col = st.selectbox("Select target column", options=all_candidates, index=default_idx)
run = st.button("⚡  Train Baseline Models", use_container_width=True)

if run and target_col:
    with st.spinner("Training models..."):
        try:
            result = train_models(df, target_col, id_cols)
            st.session_state["train_result"] = result
            # Also merge into vektor_data for context builder
            st.session_state["vektor_data"].update({
                "train_results":   result["results"],
                "train_task":      result["task"],
                "train_target":    result["target"],
                "train_leaky_cols":result["leaky_cols"],
                "train_cv_folds":  result["cv_folds"],
            })
        except Exception as e:
            st.markdown(f'<div class="alert-warn">⚠ Training failed: {e}</div>', unsafe_allow_html=True)

if "train_result" not in st.session_state:
    st.stop()

res      = st.session_state["train_result"]
results  = res["results"]
task     = res["task"]
leaky    = res["leaky_cols"]
cv_folds = res["cv_folds"]

def rank_key(k):
    m = results[k]
    return m["cv_mean"] if m.get("cv_mean") is not None else m["primary"]

best_model = max(results, key=rank_key)
has_cv     = any(results[k].get("cv_mean") is not None for k in results)

if leaky:
    st.markdown(f'<div class="alert-warn">⚠ Excluded <strong>{", ".join(leaky)}</strong> — leaky features.</div>', unsafe_allow_html=True)

summary = (f'<div class="alert-info">✅ Trained <strong>{len(results)} models</strong> on '
           f'<strong>{res["n_total"]} rows</strong> using <strong>{res["n_features"]} features</strong> · '
           f'Target: <strong>{target_col}</strong> · Test set: <strong>{res["n_test"]} rows</strong> · '
           f'Task: <strong>{task.title()}</strong></div>')
st.markdown(summary, unsafe_allow_html=True)

best_primary = results[best_model]["primary"]
best_cv = results[best_model].get("cv_mean")
if best_primary >= 97 and res["n_total"] < 2000 and best_cv is not None:
    if abs(best_primary - best_cv) > 5:
        st.markdown(f'<div class="alert-warn">⚠ Train/test score ({best_primary}%) is notably higher than {cv_folds}-fold CV ({best_cv}%) — single split was optimistic. Rankings use CV.</div>', unsafe_allow_html=True)

# Model cards
st.markdown('<p class="section-header">Model Results</p>', unsafe_allow_html=True)
cols = st.columns(len(results))
for i, (name, metrics) in enumerate(results.items()):
    is_best = name == best_model
    card_cls = "model-card model-card-best" if is_best else "model-card"
    with cols[i]:
        mhtml = ""
        for k, v in metrics.items():
            if k in ("primary","cv_mean","cv_std"): continue
            unit = "%" if k in ["Accuracy","F1 Score","R² Score"] else ""
            mhtml += f'<div class="model-metric-row"><span class="model-metric-label">{k}</span><span class="model-metric-value">{v}{unit}</span></div>'
        cv_m, cv_s = metrics.get("cv_mean"), metrics.get("cv_std")
        if cv_m is not None:
            mhtml += f'<div class="model-metric-row"><span class="model-metric-label">{cv_folds}-Fold CV</span><span class="model-metric-value">{cv_m}% <span style="font-size:0.65rem;color:#5a5a8a;">±{cv_s}</span></span></div>'
        st.markdown(f'<div class="{card_cls}"><div class="model-name">{name}</div>{mhtml}</div>', unsafe_allow_html=True)

# Comparison bars
primary_metric = "Accuracy" if task == "classification" else "R² Score"
st.markdown(f'<p class="section-header">Model Comparison {"(ranked by CV)" if has_cv else ""}</p>', unsafe_allow_html=True)
colors = ["#7c5cf8","#a855f7","#5dbb6a"]
comp_html = ""
for idx, (name, metrics) in enumerate(sorted(results.items(), key=lambda x: rank_key(x[0]), reverse=True)):
    val = metrics.get("cv_mean") or metrics.get("primary", 0)
    std = metrics.get("cv_std", "")
    bold = "font-weight:700;color:#c8c8f0;" if name == best_model else ""
    std_label = f' <span style="font-size:0.65rem;color:#4a4a6a;">±{std}</span>' if std else ""
    comp_html += (f'<div class="compare-row">'
                  f'<div class="compare-model-name" style="{bold}">{name}</div>'
                  f'<div class="compare-bar-wrap"><div class="compare-bar-fill" style="width:{max(val,2)}%;background:{colors[idx%3]};"></div></div>'
                  f'<div class="compare-val">{val}%{std_label}</div></div>')
label = f"{cv_folds}-Fold CV {primary_metric}" if has_cv else primary_metric
st.markdown(f'<div style="background:#0a0a18;border:1px solid #14143a;border-radius:12px;padding:1.5rem 1.8rem;"><div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#4a4a6a;margin-bottom:1rem;font-weight:600;">{label}</div>{comp_html}</div>', unsafe_allow_html=True)

# Winner
wm = results[best_model]
cv_v, cv_s2 = wm.get("cv_mean"), wm.get("cv_std")
score_line = (f'{cv_folds}-Fold CV: <strong style="color:#a855f7">{cv_v}% ±{cv_s2}</strong> · Single-split: {wm["primary"]}%'
              + (' · <span style="color:#e8934a;">high variance — treat with caution</span>' if cv_s2 and cv_s2 > 15 else "")
              if cv_v else f'{primary_metric}: <strong style="color:#a855f7">{wm["primary"]}%</strong>')
st.markdown(f'<div style="background:linear-gradient(135deg,#0f0d22,#12103a);border:1px solid #3a2a6a;border-radius:14px;padding:1.5rem 2rem;margin-top:1rem;display:flex;align-items:center;gap:1.5rem;"><div style="font-size:2.5rem;">🏆</div><div><div style="font-size:0.6rem;letter-spacing:3px;text-transform:uppercase;color:#5a5a8a;font-weight:600;">Best Model {"(by CV)" if cv_v else ""}</div><div style="font-size:1.4rem;font-weight:700;color:#c4b5fd;margin:0.2rem 0;">{best_model}</div><div style="font-size:0.8rem;color:#6060a0;">{score_line}</div></div></div>', unsafe_allow_html=True)
