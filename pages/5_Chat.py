"""Page 5 — Chat with Your Dataset"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from modules.groq_client import chat_turn
from modules.context_builder import build_dataset_context, build_chat_system_prompt

st.set_page_config(page_title="Vektor · Chat", page_icon="💬", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "vektor_data" not in st.session_state:
    st.warning("⚠ Please upload a dataset on the Home page first.")
    st.stop()

d   = st.session_state["vektor_data"]
df  = d["df"]
api_key = st.session_state.get("groq_api_key_saved", "")

st.markdown('<p class="section-header">Chat with Your Dataset</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#3a3a5a;font-size:0.78rem;margin:-0.5rem 0 1rem 0;">Ask anything about your data — answers are grounded in real computed facts.</p>', unsafe_allow_html=True)

if not api_key:
    st.markdown('<div class="alert-info">ℹ Enter your Groq API key on the <strong>AI Narrative</strong> page first.</div>', unsafe_allow_html=True)
    st.stop()

# Build context once per session
if "chat_context" not in st.session_state:
    context = build_dataset_context(df, d)
    st.session_state["chat_context"] = context
    st.session_state["chat_system"] = build_chat_system_prompt(context)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── Suggestion chips ──────────────────────────────────────────────────────────
suggestions = [
    "What's the biggest data quality issue?",
    "Which column has the most signal?",
    "Is this dataset ready for ML?",
    "What should I do before training?",
    "Which model performed best and why?",
    "What type of ML problem is this?",
]
s_cols = st.columns(3)
for i, q in enumerate(suggestions):
    with s_cols[i % 3]:
        if st.button(q, key=f"sug_{i}", use_container_width=True):
            st.session_state["pending_input"] = q

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Ask about your data...")
active = pending or user_input

if active:
    st.session_state["chat_history"].append({"role": "user", "content": active})
    with st.chat_message("user"):
        st.markdown(active)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chat_turn(api_key, st.session_state["chat_system"],
                                   st.session_state["chat_history"])
                st.markdown(answer)
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Chat error: {e}")
                st.session_state["chat_history"].pop()

# ── Clear ─────────────────────────────────────────────────────────────────────
if st.session_state["chat_history"]:
    if st.button("🗑️ Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()
