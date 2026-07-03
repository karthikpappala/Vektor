"""Builds grounded context strings for LLM prompts from Vektor's computed facts."""
import pandas as pd
from modules.utils import is_id_column


def build_dataset_context(df, meta: dict) -> str:
    """Compact factual summary injected into every LLM prompt."""
    parts = []
    parts.append(f"Dataset: {meta['n_rows']} rows × {meta['n_cols']} columns. Quality: {meta['qs']}/100.")
    parts.append(f"Missing values: {meta['n_nulls']} total ({df.isnull().mean().mean()*100:.1f}% of cells). Duplicates: {meta['n_dupes']}.")

    if meta.get("id_cols"):
        parts.append(f"ID columns (excluded): {', '.join(meta['id_cols'])}.")
    if meta.get("const_cols"):
        parts.append(f"Constant columns: {', '.join(meta['const_cols'])}.")
    if meta.get("high_null"):
        parts.append(f"High-null columns (>30%): {', '.join(meta['high_null'])}.")
    if meta.get("high_skew"):
        parts.append(f"Highly skewed columns: {', '.join(meta['high_skew'])}.")
    if meta.get("strong_corr"):
        pairs = "; ".join([f"{a}&{b}(r={v})" for a, b, v in meta["strong_corr"][:5]])
        parts.append(f"Strong correlations: {pairs}.")

    parts.append(f"Problem type: {meta['problem_type']}. Reason: {meta['problem_reason']}")
    parts.append(f"ML readiness: {meta['ml_score']}/100 ({meta['ml_verdict']}).")
    bd = ", ".join([f"{k}={v}" for k, v in meta["score_breakdown"].items()])
    parts.append(f"Readiness breakdown: {bd}.")

    if meta.get("feat_importance"):
        fi = ", ".join([f"{c}({v})" for c, v in meta["feat_importance"][:8]])
        parts.append(f"Feature signal strengths: {fi}.")

    # Per-column snapshot (capped at 20 columns)
    col_stats = []
    for col in df.columns[:20]:
        null_pct = round(df[col].isnull().mean() * 100, 1)
        nuniq    = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]) and not is_id_column(df[col], col):
            try:
                mn = round(float(df[col].mean()), 3) if not df[col].isnull().all() else "N/A"
                mx = round(float(df[col].max()),  3) if not df[col].isnull().all() else "N/A"
            except (TypeError, ValueError):
                mn, mx = "N/A", "N/A"
            col_stats.append(f"{col}(num,null={null_pct}%,u={nuniq},mean={mn},max={mx})")
        else:
            top = str(df[col].value_counts().index[0])[:15] if nuniq > 0 else "N/A"
            col_stats.append(f"{col}(cat,null={null_pct}%,u={nuniq},top='{top}')")
    parts.append("Columns: " + "; ".join(col_stats) + ".")

    if meta.get("train_results"):
        tr     = meta["train_results"]
        tgt    = meta.get("train_target", "")
        task   = meta.get("train_task", "")
        leaky  = meta.get("train_leaky_cols", [])
        cv_k   = meta.get("train_cv_folds", 5)
        mstrs  = []
        for nm, m in tr.items():
            cv, cvs, prim = m.get("cv_mean"), m.get("cv_std"), m.get("primary")
            mstrs.append(f"{nm}: CV={cv}%±{cvs}, single-split={prim}%" if cv else f"{nm}: {prim}%")
        parts.append(f"Models (target='{tgt}', {task}): " + "; ".join(mstrs))
        if leaky:
            parts.append(f"Leaky columns excluded: {leaky}.")

    return "\n".join(parts)


def build_chat_system_prompt(context: str) -> str:
    return (
        "You are Vektor, an expert data analyst AI. "
        "Answer questions about the user's dataset concisely and specifically, "
        "grounding your answers in the provided facts only. "
        "Do NOT invent column names, statistics, or relationships not in the facts. "
        "If asked something you cannot compute, say so and suggest what to do. "
        "Keep answers under 120 words unless more is clearly needed. "
        "No markdown headers — plain prose or short bullets.\n\n"
        f"=== DATASET FACTS ===\n{context}\n=== END FACTS ==="
    )
