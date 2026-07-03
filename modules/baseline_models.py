"""Baseline model training — pure logic, no Streamlit."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    lgb.basic._log_warning = lambda *a, **k: None
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def detect_leaky_cols(df, feature_cols, target_col):
    leaky = []
    target_clean = df[target_col].astype(str)
    for col in feature_cols:
        grouped = pd.DataFrame({"f": df[col].astype(str), "t": target_clean}).dropna()
        if grouped.empty or grouped["f"].nunique() < 2:
            continue
        purity = grouped.groupby("f")["t"].nunique()
        hard = (purity == 1).mean() > 0.98 and grouped["f"].nunique() >= grouped["t"].nunique()
        def majority(s): return s.value_counts(normalize=True).iloc[0]
        wp = grouped.groupby("f")["t"].apply(majority)
        gs = grouped.groupby("f").size()
        soft_score = (wp * gs).sum() / len(grouped)
        soft = soft_score > 0.92 and grouped["f"].nunique() <= 30
        if hard or soft:
            leaky.append(col)
    return leaky


def prepare_features(df, feature_cols, target_col):
    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()
    mask  = y_raw.notna()
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    for col in X_raw.columns:
        if X_raw[col].dtype == object:
            X_raw[col] = X_raw[col].fillna("__missing__")
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        else:
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())
    return X_raw, y_raw


def train_models(df, target_col, id_cols):
    all_candidates  = [c for c in df.columns if c not in id_cols and df[c].nunique() > 1]
    feature_cols    = [c for c in all_candidates if c != target_col]
    feature_cols    = [c for c in feature_cols if df[c].isnull().mean() < 0.5]
    leaky_cols      = detect_leaky_cols(df, feature_cols, target_col)
    feature_cols    = [c for c in feature_cols if c not in leaky_cols]

    X, y_raw = prepare_features(df, feature_cols, target_col)
    n_unique  = y_raw.nunique()
    is_clf    = y_raw.dtype == object or n_unique <= 20
    task      = "classification" if is_clf else "regression"

    if is_clf:
        le = LabelEncoder()
        y  = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values.astype(float)

    if len(X) < 10:
        raise ValueError("Not enough rows to train (need ≥ 10 clean rows).")

    test_size = 0.2 if len(X) >= 50 else 0.3
    try:
        stratify = y if (is_clf and n_unique <= 20 and min(np.bincount(y)) >= 2) else None
    except Exception:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify)

    if is_clf:
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
            "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                             ("m", LogisticRegression(max_iter=1000, random_state=42))]),
        }
        if HAS_LGB:
            models["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
            "Ridge Regression": Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
        }
        if HAS_LGB:
            models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

    cv_folds  = min(5, max(2, min(np.bincount(y)) if is_clf else 5))
    cv_score  = "accuracy" if is_clf else "r2"
    results   = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            cv = cross_val_score(model, X, y, cv=cv_folds, scoring=cv_score, n_jobs=1)
            cv_mean, cv_std = round(cv.mean() * 100, 1), round(cv.std() * 100, 1)
        except Exception:
            cv_mean = cv_std = None
        if is_clf:
            avg = "binary" if n_unique == 2 else "weighted"
            acc = round(accuracy_score(y_test, y_pred) * 100, 1)
            f1  = round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 1)
            results[name] = {"Accuracy": acc, "F1 Score": f1,
                              "primary": acc, "cv_mean": cv_mean, "cv_std": cv_std}
        else:
            rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            r2   = round(r2_score(y_test, y_pred) * 100, 1)
            results[name] = {"R² Score": r2, "RMSE": rmse,
                              "primary": r2, "cv_mean": cv_mean, "cv_std": cv_std}

    return {
        "results": results, "task": task, "target": target_col,
        "n_total": len(X), "n_features": len(feature_cols),
        "n_test": len(X_test), "cv_folds": cv_folds,
        "leaky_cols": leaky_cols, "feature_cols": feature_cols,
    }
