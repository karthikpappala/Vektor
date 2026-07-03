"""Page 4 — AI Narrative via Groq"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from modules.groq_client import generate_narrative
from modules.context_builder import build_dataset_context

st.set_page_config(page_title="Vektor · Narrative", page_icon="🧠", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "vektor_data" not in st.session_state:
    st.warning("⚠ Please upload a dataset on the Home page first.")
    st.stop()

d  = st.session_state["vektor_data"]
df = d["df"]

st.markdown('<p class="section-header">AI Narrative</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#3a3a5a;font-size:0.78rem;margin:-0.5rem 0 1rem 0;">A plain-English story of your dataset — grounded in real computed facts.</p>', unsafe_allow_html=True)

# ── API key ───────────────────────────────────────────────────────────────────
api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...",
                         key="groq_api_key")
if api_key:
    st.session_state["groq_api_key_saved"] = api_key
else:
    api_key = st.session_state.get("groq_api_key_saved", "")

st.markdown('<p style="color:#3a3a5a;font-size:0.72rem;">Session-only — never stored. Get a free key at <a href="https://console.groq.com/keys" target="_blank" style="color:#7c5cf8;">console.groq.com/keys</a></p>', unsafe_allow_html=True)

generate = st.button("✨  Generate AI Narrative", use_container_width=True, disabled=not api_key)

if not api_key:
    st.markdown('<div class="alert-info">ℹ Enter your Groq API key above to generate a narrative.</div>', unsafe_allow_html=True)

if generate and api_key:
    with st.spinner("Writing narrative..."):
        try:
            context = build_dataset_context(df, d)
            narrative = generate_narrative(api_key, context)
            st.session_state["ai_narrative"] = narrative
        except Exception as e:
            st.markdown(f'<div class="alert-warn">⚠ Failed: {e}</div>', unsafe_allow_html=True)

if "ai_narrative" in st.session_state:
    st.markdown(f"""<div class="narrative-wrap">
        <div class="narrative-badge">✨ AI-Generated &nbsp;·&nbsp; Llama 3.1 via Groq</div>
        <div class="narrative-text">{st.session_state['ai_narrative']}</div>
    </div>""", unsafe_allow_html=True)
