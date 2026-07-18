from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Set common style for plots
STYLE = "seaborn-v0_8-whitegrid"
plt.style.use(STYLE)
plt.rcParams.update({
    "savefig.dpi":       200,
    "font.size":         15,
    "axes.titlesize":    17,
    "axes.titleweight":  "bold",
    "axes.labelsize":    16,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "legend.fontsize":   13,
    "legend.title_fontsize": 13,
    "figure.titlesize":  18,
})

METHOD_COLORS = {
    "real":             "#2c3e50",
    "ctgan":            "#e74c3c",
    "tvae":             "#3498db",
    "gaussian_copula":  "#2ecc71",
    "smote":            "#f39c12",
}



def plot_marginal_distributions(
    real_df: pd.DataFrame,
    synthetic_dfs: dict[str, pd.DataFrame],
    col: str,
    cfg: dict[str, Any],
    save: bool = True,
) -> plt.Figure:
    """Distribution of one column, real vs each synthetic dataset."""
    cat_cols = set(cfg["dataset"].get("categorical_columns", []))
    is_cat   = col in cat_cols or real_df[col].dtype == object

    all_dfs = {"real": real_df} | synthetic_dfs
    n_methods = len(all_dfs)
    fig, axes = plt.subplots(1, n_methods, figsize=(3.2 * n_methods, 3.6), sharey=False)
    if n_methods == 1:
        axes = [axes]

    for ax, (method, df) in zip(axes, all_dfs.items()):
        color = METHOD_COLORS.get(method, "grey")
        series = df[col].dropna()

        if is_cat:
            vc = series.value_counts(normalize=True).head(10)
            ax.bar(range(len(vc)), vc.values, color=color, alpha=0.85)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=45, ha="right", fontsize=12)
            ax.set_ylabel("Relative frequency")
        else:
            ax.hist(series, bins=30, color=color, alpha=0.75, density=True, edgecolor="white")
            ax.set_ylabel("Density")

        ax.set_title(method, fontsize=15)
        ax.set_xlabel(col)

    fig.suptitle(f"Marginal distribution: {col}", y=1.02)
    plt.tight_layout()

    if save:
        _save_fig(fig, f"marginal_{col}", cfg)

    return fig


def plot_fidelity_summary(
    reports: list[dict[str, Any]],
    cfg: dict[str, Any],
    save: bool = True,
) -> plt.Figure:
    """MMD and correlation MAE per method, side by side."""
    methods      = [r["method"] for r in reports]
    mmd_values   = [r["mmd"] for r in reports]
    corr_values  = [r["correlation_mae"] for r in reports]
    colors       = [METHOD_COLORS.get(m, "grey") for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    _bar_chart(ax1, methods, mmd_values, colors, "MMD (RBF)", "lower is better")
    _bar_chart(ax2, methods, corr_values, colors, "Correlation MAE", "lower is better")

    fig.suptitle("Fidelity: real vs synthetic")
    plt.tight_layout()

    if save:
        _save_fig(fig, "fidelity_summary", cfg)

    return fig


def plot_class_balance(
    real_df: pd.DataFrame,
    synthetic_dfs: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    save: bool = True,
) -> plt.Figure:
    """Target class distribution, real vs synthetic."""
    target = cfg["dataset"]["target_column"]
    all_dfs = {"real": real_df} | synthetic_dfs

    methods = list(all_dfs.keys())
    class_labels = sorted(real_df[target].unique())
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))

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
                ha="center", va="bottom", fontsize=12,
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
    save: bool = True,
    filename_suffix: str = "",
) -> plt.Figure:
    """Spearman correlation matrix of the numerical columns as a heatmap."""
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr(method="spearman")
    n      = len(corr)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.55)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=11)
    ax.set_yticklabels(corr.columns, fontsize=11)
    ax.set_title(title, fontweight="bold", pad=12)

    plt.tight_layout()

    if save:
        _save_fig(fig, f"correlation_heatmap{filename_suffix}", cfg)

    return fig


def plot_utility_bar(
    metrics_df: pd.DataFrame,
    metric: str,
    cfg: dict[str, Any],
    save: bool = True,
) -> plt.Figure:
    """Grouped bars of one utility metric, per classifier and method."""
    colors      = METHOD_COLORS
    classifiers = metrics_df["classifier"].unique().tolist()
    methods     = [cfg["experiments"]["baseline_label"]] + list(cfg["generation"]["methods"])
    methods     = [m for m in methods if m in metrics_df["method"].unique()]

    x       = np.arange(len(classifiers))
    n_meth  = len(methods)
    width   = 0.7 / n_meth
    offsets = np.linspace(-(n_meth - 1) / 2, (n_meth - 1) / 2, n_meth) * width

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, method in enumerate(methods):
        sub    = metrics_df[metrics_df["method"] == method].set_index("classifier")
        values = [sub.loc[c, metric] if c in sub.index else float("nan") for c in classifiers]
        bars   = ax.bar(
            x + offsets[i], values, width,
            label=method,
            color=colors.get(method, "grey"),
            alpha=0.85,
            edgecolor="white",
        )
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()}: TSTR vs real baseline", fontweight="bold")
    ax.legend(title="Training data", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=13)
    plt.tight_layout()

    if save:
        _save_fig(fig, f"utility_{metric}", cfg)

    return fig


def plot_fairness_bar(
    metrics_df: pd.DataFrame,
    metric: str,
    cfg: dict[str, Any],
    save: bool = True,
) -> plt.Figure:
    """Grouped bars of one mean-fairness metric, per classifier and method."""
    colors      = METHOD_COLORS
    classifiers = metrics_df["classifier"].unique().tolist()
    methods     = [cfg["experiments"]["baseline_label"]] + list(cfg["generation"]["methods"])
    methods     = [m for m in methods if m in metrics_df["method"].unique()]

    x       = np.arange(len(classifiers))
    n_meth  = len(methods)
    width   = 0.7 / n_meth
    offsets = np.linspace(-(n_meth - 1) / 2, (n_meth - 1) / 2, n_meth) * width

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, method in enumerate(methods):
        sub    = metrics_df[metrics_df["method"] == method].set_index("classifier")
        values = [sub.loc[c, metric] if c in sub.index else float("nan") for c in classifiers]
        bars   = ax.bar(
            x + offsets[i], values, width,
            label=method,
            color=colors.get(method, "grey"),
            alpha=0.85,
            edgecolor="white",
        )
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()}: TSTR vs real baseline", fontweight="bold")
    ax.legend(title="Training data", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=13)
    plt.tight_layout()

    if save:
        _save_fig(fig, f"fairness_{metric}", cfg)

    return fig


def plot_delta_heatmap(
    delta_df: pd.DataFrame,
    delta_columns: list[str],
    filename_suffix: str,
    cfg: dict[str, Any],
    title: str = "Δ metrics (synthetic vs. real baseline)",
    save: bool = True,
) -> plt.Figure:
    """Heatmap of metric changes from the real baseline (red = worse, blue = better)."""
    baseline_label = cfg["experiments"]["baseline_label"]
    plot_df = delta_df[delta_df["method"] != baseline_label].copy()

    plot_df["row_label"] = plot_df["method"].str.upper() + " / " + plot_df["classifier"]
    pivot = plot_df.set_index("row_label")[delta_columns]

    col_labels = [c.replace("delta_", "Δ ").replace("_", " ") for c in delta_columns]
    figsize    = (max(7, len(delta_columns) * 1.4), max(4, len(pivot) * 0.5 + 1))

    abs_max = pivot.abs().max().max()
    abs_max = abs_max if abs_max > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 12},
        linewidths=0.4,
        linecolor="white",
        xticklabels=col_labels,
        cbar_kws={"shrink": 0.7, "label": "Δ"},
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=30, labelsize=13)
    ax.tick_params(axis="y", rotation=0,  labelsize=13)
    plt.tight_layout()

    if save:
        _save_fig(fig, f"delta_heatmap_{filename_suffix}", cfg)

    return fig


def plot_utility_fairness_scatter(
    delta_df: pd.DataFrame,
    utility_col: str,
    fairness_col: str,
    cfg: dict[str, Any],
    save: bool = True,
) -> plt.Figure:
    """Scatter of utility change vs fairness change, one point per (method, classifier)."""
    colors         = METHOD_COLORS
    baseline_label = cfg["experiments"]["baseline_label"]
    plot_df        = delta_df[delta_df["method"] != baseline_label].copy()
    classifiers    = plot_df["classifier"].unique()

    markers     = ["o", "s", "^", "D", "v"]
    clf_markers = {c: markers[i % len(markers)] for i, c in enumerate(classifiers)}

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    for method in cfg["generation"]["methods"]:
        sub = plot_df[plot_df["method"] == method]
        for clf in classifiers:
            row = sub[sub["classifier"] == clf]
            if row.empty:
                continue
            ax.scatter(
                row[utility_col].values,
                row[fairness_col].values,
                color=colors.get(method, "grey"),
                marker=clf_markers[clf],
                s=130,
                label=f"{method} / {clf}",
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel(utility_col.replace("delta_", "Δ ").replace("_", " ").title())
    ax.set_ylabel(fairness_col.replace("delta_", "Δ ").replace("_", " ").title())
    ax.set_title("Utility loss vs. Fairness change (relative to real baseline)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=12)
    plt.tight_layout()

    if save:
        _save_fig(fig, "scatter_utility_vs_fairness", cfg)

    return fig


def plot_per_attribute_fairness(
    fairness_df: pd.DataFrame,
    metric: str,
    cfg: dict[str, Any],
    title: str = "",
    save: bool = True,
    filename_suffix: str = "",
) -> plt.Figure:
    """Horizontal bar chart of one fairness metric per protected attribute, red when it fails the acceptable threshold (DPD/EOD > 0.1, DI < 0.8) and green otherwise."""
    fairness_df = fairness_df.loc[:, ~fairness_df.columns.duplicated()]
    sub = fairness_df[["attribute", metric]].dropna(subset=[metric])
    sub = sub.sort_values(metric, ascending=(metric != "di"))

    colors = [
        "#e74c3c" if (metric != "di" and v > 0.1) or (metric == "di" and v < 0.8)
        else "#2ecc71"
        for v in sub[metric]
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(sub["attribute"], sub[metric], color=colors, alpha=0.85, edgecolor="white")

    # Dashed line at the metric's acceptability threshold; green side is fair.
    if metric == "di":
        ax.axvline(0.8, color="darkorange", lw=1.2, ls="--", label="4/5 rule: DI >= 0.8 is fair")
    else:
        ax.axvline(0.1, color="darkorange", lw=1.2, ls="--", label=f"{metric.upper()} <= 0.1 is fair")
    ax.legend(fontsize=13)

    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} per Protected Attribute", fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    plt.tight_layout()

    if save:
        _save_fig(fig, f"per_attr_{metric}{filename_suffix}", cfg)

    return fig


def plot_mmd_bar(
    metrics_df: pd.DataFrame,
    cfg: dict[str, Any],
    save: bool = True,
) -> plt.Figure:
    """Bar chart of MMD per generation method."""
    colors         = METHOD_COLORS
    baseline_label = cfg["experiments"]["baseline_label"]

    mmd_per_method = (
        metrics_df.groupby("method")["mmd"].first().reset_index()
        if "mmd" in metrics_df.columns
        else pd.DataFrame(columns=["method", "mmd"])
    )
    mmd_per_method = mmd_per_method[mmd_per_method["method"] != baseline_label]
    mmd_per_method = mmd_per_method.sort_values("mmd")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        mmd_per_method["method"],
        mmd_per_method["mmd"],
        color=[colors.get(m, "grey") for m in mmd_per_method["method"]],
        alpha=0.85,
        edgecolor="white",
    )
    for bar, v in zip(bars, mmd_per_method["mmd"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mmd_per_method["mmd"].max() * 0.01, 1e-5),
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=12,
        )

    ax.set_ylabel("MMD^2 (RBF kernel)")
    ax.set_xlabel("Generation method")
    ax.set_title("MMD: synthetic vs real features", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()

    if save:
        _save_fig(fig, "mmd_comparison", cfg)

    return fig



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
            ha="center", va="bottom", fontsize=12,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\n{subtitle}", fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")


def _save_fig(fig: plt.Figure, name: str, cfg: dict[str, Any]) -> None:
    figures_dir = cfg["paths"]["figures_dir"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(figures_dir) / f"{name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Figure saved -> {out_path}")