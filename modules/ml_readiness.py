"""ML Readiness computations — pure logic, no Streamlit."""
import numpy as np
import pandas as pd


def detect_problem_type(df, cat_cols, anal_num_cols, id_cols):
    text_cols = [c for c in cat_cols if df[c].dropna().str.len().mean() > 80]
    if text_cols:
        return ("NLP / Text Classification",
                f"Detected {len(text_cols)} long-text column(s): {', '.join(text_cols[:2])}.",
                "🔤", text_cols)

    date_like = [c for c in df.columns if any(kw in c.lower()
                 for kw in ["date","time","year","month","day","timestamp"])]
    if date_like and anal_num_cols:
        return ("Time Series Forecasting",
                f"Found temporal column(s): {', '.join(date_like[:2])}. Suited for ARIMA / LSTM.",
                "📈", [])

    for col in cat_cols:
        if df[col].nunique() == 2:
            return ("Binary Classification",
                    f"Column '{col}' has 2 unique values — likely your target.",
                    "🎯", [c for c in df.columns if c != col and c not in id_cols])

    for col in cat_cols:
        if 2 < df[col].nunique() <= 20:
            return ("Multi-class Classification",
                    f"Column '{col}' has {df[col].nunique()} categories — likely your target.",
                    "🏷️", [c for c in df.columns if c != col and c not in id_cols])

    if anal_num_cols:
        return ("Regression",
                f"Mostly numeric. '{anal_num_cols[-1]}' may be your target.",
                "📊", anal_num_cols[:-1])

    return ("Unsupervised / Clustering",
            "No obvious target detected. Suits K-Means or DBSCAN.",
            "🔵", [])


def compute_ml_readiness(df, anal_num_cols, cat_cols, const_cols, id_cols):
    scores = {}
    scores["Completeness"]    = max(0, round((1 - df.isnull().mean().mean() * 2) * 100))
    n_features                = max(len(anal_num_cols) + len(cat_cols), 1)
    scores["Size Adequacy"]   = min(100, round((len(df) / n_features) * 3))
    bad                       = len(const_cols) + len(id_cols)
    scores["Feature Quality"] = max(0, round((1 - bad / max(len(df.columns), 1)) * 100))
    if anal_num_cols:
        hs = sum(1 for c in anal_num_cols if abs(df[c].skew()) > 1.5)
        scores["Numeric Health"] = max(0, round((1 - hs / len(anal_num_cols)) * 100))
    else:
        scores["Numeric Health"] = 50
    has_target = any(any(kw in c.lower() for kw in
                    ["label","target","class","category","score","result","outcome","status","type"])
                    for c in df.columns)
    scores["Label Readiness"] = 90 if has_target else 45
    return round(sum(scores.values()) / len(scores)), scores


def get_feature_importance(df, anal_num_cols, cat_cols):
    scores = {}
    for col in anal_num_cols:
        data = df[col].dropna()
        cv = abs(data.std() / data.mean()) if data.mean() != 0 else data.std()
        scores[col] = min(cv, 5.0)
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True)
        entropy = -(counts * np.log(counts + 1e-9)).sum()
        max_e = np.log(max(len(counts), 2))
        scores[col] = entropy / max_e if max_e > 0 else 0
    if not scores:
        return []
    max_s = max(scores.values()) or 1
    return [(c, round(v / max_s * 100, 1))
            for c, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:12]]


def get_preprocessing_steps(df, anal_num_cols, cat_cols, const_cols, id_cols, problem_type):
    steps = []
    if const_cols:
        steps.append(("high","🗑️","Drop Constant Columns",
                       f"Remove: {', '.join(const_cols[:3])}."))
    if id_cols:
        steps.append(("high","🔑","Remove ID Columns",
                       f"Drop: {', '.join(id_cols)}."))
    high_miss = [c for c in df.columns if df[c].isnull().mean() > 0.3]
    miss_cols = [c for c in df.columns if 0 < df[c].isnull().mean() <= 0.3]
    if high_miss:
        steps.append(("high","⚠️","Drop High-Null Columns",
                       f"Drop or impute: {', '.join(high_miss[:3])} (>30% missing)."))
    if miss_cols:
        steps.append(("medium","🔧","Impute Missing Values",
                       f"Fill nulls in {', '.join(miss_cols[:3])} with median/mode."))
    high_skew = [c for c in anal_num_cols if abs(df[c].skew()) > 1.5]
    if high_skew:
        steps.append(("medium","📐","Fix Skewed Distributions",
                       f"Apply log1p to: {', '.join(high_skew[:3])}."))
    if anal_num_cols:
        steps.append(("medium","⚖️","Scale Numeric Features",
                       "Apply StandardScaler before training."))
    high_card = [c for c in cat_cols if df[c].nunique() > 50]
    low_card  = [c for c in cat_cols if 2 <= df[c].nunique() <= 50]
    if high_card:
        steps.append(("medium","🏷️","Encode High-Cardinality Cols",
                       f"Target encode: {', '.join(high_card[:3])}."))
    if low_card:
        steps.append(("low","🔤","One-Hot Encode Categoricals",
                       f"pd.get_dummies() on: {', '.join(low_card[:3])}."))
    if df.duplicated().sum() > 0:
        steps.append(("high","🔁","Remove Duplicates",
                       f"Drop {df.duplicated().sum()} duplicate rows."))
    if "NLP" in problem_type:
        steps.append(("high","📝","Preprocess Text",
                       "Lowercase, tokenise, apply TF-IDF or embeddings."))
    if "Time Series" in problem_type:
        steps.append(("high","📅","Parse Datetime",
                       "pd.to_datetime(), sort, engineer lag features."))
    if not steps:
        steps.append(("low","✅","Dataset Looks Clean",
                       "No critical issues. Proceed to feature engineering."))
    return steps
