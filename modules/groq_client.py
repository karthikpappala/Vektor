"""Groq API client — handles both narrative and chat requests."""
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS_BASE = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


def _post(api_key: str, messages: list, temperature=0.35, max_tokens=500) -> str:
    headers = {**HEADERS_BASE, "Authorization": f"Bearer {api_key}"}
    resp = requests.post(
        GROQ_URL,
        json={
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_narrative(api_key: str, context_summary: str) -> str:
    system = (
        "You are a senior data analyst writing a short narrative briefing for a colleague. "
        "You have been given pre-computed findings about a dataset. Write 3-4 short paragraphs "
        "in plain English covering: what the data looks like, what stands out, risks or caveats, "
        "and what to do next. Reference actual numbers. Do NOT invent any facts. "
        "Do not use markdown headers or bullet points — write in flowing prose. "
        "Keep total length under 220 words. "
        "CRITICAL: when model results list both a cross-validated score and a single-split score, "
        "report ONLY the cross-validated score as the model's performance. "
        "If unsure, use the smaller, more conservative number."
    )
    return _post(api_key, [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Findings:\n\n{context_summary}\n\nWrite the briefing."},
    ], temperature=0.4, max_tokens=500)


def chat_turn(api_key: str, system_prompt: str, history: list) -> str:
    messages = [{"role": "system", "content": system_prompt}] + history
    return _post(api_key, messages, temperature=0.3, max_tokens=350)
