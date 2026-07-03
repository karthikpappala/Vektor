"""EDA computations — pure logic, no Streamlit."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules.utils import skewness_label, get_outlier_count


def get_strong_correlations(df: pd.DataFrame, cols: list, threshold=0.8) -> list:
    corr = df[cols].corr().abs()
    strong = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if v > threshold:
                strong.append((cols[i], cols[j], round(v, 3)))
    return strong


def get_quality_alerts(df, anal_num_cols, cat_cols, const_cols, id_cols) -> dict:
    return {
        "high_null":  [c for c in df.columns if df[c].isnull().mean() > 0.3],
        "high_skew":  [c for c in anal_num_cols if abs(df[c].skew()) > 1.5],
        "high_card":  [c for c in cat_cols if df[c].nunique() > 50],
        "duplicates": int(df.duplicated().sum()),
        "const_cols": const_cols,
        "id_cols":    id_cols,
    }


def plot_distributions(df: pd.DataFrame, cols: list):
    n = min(len(cols), 8)
    cols = cols[:n]
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 2.6), squeeze=False)
    fig.patch.set_facecolor("#07070f")
    axes_flat = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes_flat[i]
        ax.set_facecolor("#0d0d1f")
        _, _, patches = ax.hist(df[col].dropna(), bins=25,
                                color="#7c5cf8", alpha=0.9,
                                edgecolor="#07070f", linewidth=0.4)
        for j, patch in enumerate(patches):
            frac = j / max(len(patches) - 1, 1)
            r = int(0x7c + (0xc0 - 0x7c) * frac)
            g = int(0x5c + (0x84 - 0x5c) * frac)
            b = int(0xf8 + (0xfc - 0xf8) * frac)
            patch.set_facecolor(f"#{r:02x}{g:02x}{b:02x}")
        ax.set_title(col, color="#c8c8e8", fontsize=8.5, fontweight="600", pad=5)
        ax.tick_params(colors="#3a3a5a", labelsize=6.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a1a35")
    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout(pad=1.2)
    return fig


def plot_correlation(df: pd.DataFrame, cols: list):
    corr = df[cols].corr()
    size = max(6, len(cols) * 0.9)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    fig.patch.set_facecolor("#07070f")
    ax.set_facecolor("#0d0d1f")
    cmap = sns.diverging_palette(250, 10, s=80, l=40, as_cmap=True)
    sns.heatmap(corr, ax=ax, cmap=cmap, center=0,
                annot=len(cols) <= 10, fmt=".2f",
                linewidths=0.8, linecolor="#07070f",
                annot_kws={"size": 8, "color": "#e2e2f0", "weight": "600"},
                cbar_kws={"shrink": 0.7, "aspect": 20})
    ax.tick_params(colors="#6060a0", labelsize=8)
    plt.tight_layout()
    return fig
