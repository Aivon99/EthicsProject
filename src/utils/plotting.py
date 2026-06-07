from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)



# ---- Colour palette ----------------------------------------------------------
METHOD_COLORS = {
    "real":             "#2c3e50",
    "ctgan":            "#e74c3c",
    "tvae":             "#3498db",
    "gaussian_copula":  "#2ecc71",
    "smote":            "#f39c12",
}



# ---- Public functions --------------------------------------------------------
def plot_marginal_distributions(
    real_df: pd.DataFrame,
    synthetic_dfs: dict[str, pd.DataFrame],
    col: str,
    cfg: dict[str, Any],
    *,
    save: bool = True,
) -> plt.Figure:
    """
    Overlay marginal distributions for a single column across real data
    and all synthetic datasets.

    For numerical columns: KDE + histogram.
    For categorical columns: frequency bar charts.

    Parameters
    ----------
    real_df : pd.DataFrame
    synthetic_dfs : dict[str, pd.DataFrame]
        Keys are method names; values are the corresponding DataFrames.
    col : str
        Column to visualise.
    cfg : dict
    save : bool
        If True, save the figure to ``cfg["paths"]["figures_dir"]``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cat_cols = set(cfg["dataset"].get("categorical_columns", []))
    is_cat   = col in cat_cols or real_df[col].dtype == object

    all_dfs = {"real": real_df} | synthetic_dfs
    n_methods = len(all_dfs)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=False)
    if n_methods == 1:
        axes = [axes]

    for ax, (method, df) in zip(axes, all_dfs.items()):
        color = METHOD_COLORS.get(method, "grey")
        series = df[col].dropna()

        if is_cat:
            vc = series.value_counts(normalize=True).head(10)
            ax.bar(range(len(vc)), vc.values, color=color, alpha=0.85)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Relative frequency")
        else:
            ax.hist(series, bins=30, color=color, alpha=0.75, density=True, edgecolor="white")
            ax.set_ylabel("Density")

        ax.set_title(method, fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=8)

    fig.suptitle(f"Marginal distribution - {col}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        _save_fig(fig, f"marginal_{col}", cfg)

    return fig


def plot_fidelity_summary(
    reports: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing MMD and Spearman correlation MAE across all methods.

    Parameters
    ----------
    reports : list[dict]
        List of fidelity report dicts returned by ``compute_fidelity_report``.
    cfg : dict
    save : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    methods      = [r["method"] for r in reports]
    mmd_values   = [r["mmd"] for r in reports]
    corr_values  = [r["correlation_mae"] for r in reports]
    colors       = [METHOD_COLORS.get(m, "grey") for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _bar_chart(ax1, methods, mmd_values, colors, "MMD (RBF)", "(↓) lower is better")
    _bar_chart(ax2, methods, corr_values, colors, "Correlation MAE (Spearman)", "(↓) lower is better")

    fig.suptitle("Fidelity Summary: Real vs Synthetic", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _save_fig(fig, "fidelity_summary", cfg)

    return fig


def plot_class_balance(
    real_df: pd.DataFrame,
    synthetic_dfs: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    *,
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart showing target class distribution across real and
    all synthetic datasets.

    Parameters
    ----------
    real_df : pd.DataFrame
    synthetic_dfs : dict[str, pd.DataFrame]
    cfg : dict
    save : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    target = cfg["dataset"]["target_column"]
    all_dfs = {"real": real_df} | synthetic_dfs

    methods = list(all_dfs.keys())
    class_labels = sorted(real_df[target].unique())
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cls in enumerate(class_labels):
        freqs = [
            all_dfs[m][target].value_counts(normalize=True).get(cls, 0)
            for m in methods
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, freqs, width, label=f"Class {cls}", alpha=0.85)
        for bar, freq in zip(bars, freqs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{freq:.0%}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Relative frequency")
    ax.set_title(f"Target class balance: '{target}'", fontweight="bold")
    ax.legend(title="Class")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()

    if save:
        _save_fig(fig, "class_balance", cfg)

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    cfg: dict[str, Any],
    *,
    save: bool = True,
    filename_suffix: str = "",
) -> plt.Figure:
    """
    Plot a Spearman rank correlation matrix as a heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Numerical columns only.
    title : str
        Plot title.
    cfg : dict
    save : bool
    filename_suffix : str
        Appended to the filename for disambiguation (e.g. ``"_ctgan"``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr(method="spearman")
    n      = len(corr)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.55)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ_s")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title(title, fontweight="bold", pad=12)

    plt.tight_layout()

    if save:
        _save_fig(fig, f"correlation_heatmap{filename_suffix}", cfg)

    return fig



# ---- Internal helpers --------------------------------------------------------
def _bar_chart(
    ax: plt.Axes,
    methods: list[str],
    values: list[float],
    colors: list[str],
    ylabel: str,
    subtitle: str,
) -> None:
    bars = ax.bar(methods, values, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\n{subtitle}", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")


def _save_fig(fig: plt.Figure, name: str, cfg: dict[str, Any]) -> None:
    figures_dir = cfg["paths"]["figures_dir"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(figures_dir) / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved -> {out_path}")