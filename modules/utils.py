"""Shared utilities — pure logic, no Streamlit imports."""
import pandas as pd
import numpy as np


def is_id_column(series: pd.Series, name: str) -> bool:
    name_lower = name.lower()
    id_keywords = ["id", "uid", "uuid", "key", "code", "index",
                   "no", "num", "number", "serial"]
    name_match  = any(kw in name_lower for kw in id_keywords)
    all_unique  = series.nunique() == len(series.dropna())
    monotonic   = (series.dropna().is_monotonic_increasing or
                   series.dropna().is_monotonic_decreasing)
    return (name_match and all_unique) or (all_unique and monotonic and len(series) > 10)


def get_analytical_num_cols(df: pd.DataFrame) -> list:
    return [
        col for col in df.select_dtypes(include=np.number).columns
        if df[col].nunique() > 1 and not is_id_column(df[col], col)
    ]


def compute_quality_score(df: pd.DataFrame) -> float:
    completeness = 1 - df.isnull().mean().mean()
    uniqueness   = 1 - (df.duplicated().sum() / max(len(df), 1))
    n_obj        = (df.dtypes == object).sum()
    type_score   = min(1 - (n_obj / max(len(df.columns), 1)) * 0.5, 1.0)
    return round((completeness * 0.4 + uniqueness * 0.3 + type_score * 0.3) * 100, 1)


def load_dataframe(file) -> pd.DataFrame:
    name = file.name
    if name.endswith(".csv"):
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(file, encoding=enc)
            except Exception:
                file.seek(0)
    return pd.read_excel(file)


def detect_dtype_label(series: pd.Series, col_name: str):
    if pd.api.types.is_bool_dtype(series):            return "bool",       "badge-bool"
    if pd.api.types.is_datetime64_any_dtype(series):  return "datetime",   "badge-date"
    if pd.api.types.is_numeric_dtype(series):
        if is_id_column(series, col_name):            return "id / key",   "badge-id"
        return "numeric", "badge-num"
    return "categorical", "badge-cat"


def null_class(pct: float) -> str:
    if pct > 30: return "null-high"
    if pct > 10: return "null-mid"
    return "null-low"


def skewness_label(skew: float):
    if abs(skew) < 0.5: return "✓ symmetric",     "#5dbb6a"
    if abs(skew) < 1.0: return "~ moderate skew", "#d4a040"
    return "⚠ high skew", "#e86060"


def get_outlier_count(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
