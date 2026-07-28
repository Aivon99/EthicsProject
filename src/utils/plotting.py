from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger
from src.generation.metadata import categorical_columns

logger = get_logger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "savefig.dpi": 200,
    "figure.dpi": 110,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.titlesize": 15,
})

METHOD_COLORS = {
    "real": "#2c3e50",
    "ctgan": "#e74c3c",
    "tvae": "#3498db",
    "gaussian_copula": "#2ecc71",
    "smote": "#f39c12",
    "smote_low_perf": "#f39c12",
}

GREY = "#95a5a6"


def method_order(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Methods actually present in a results frame, baseline first."""
    baseline = cfg["experiments"]["baseline_label"]
    present = list(dict.fromkeys(df["method"]))
    return [m for m in present if m == baseline] + [m for m in present if m != baseline]


def save_figure(fig: plt.Figure, name: str, figures_dir) -> Path:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{name}.png"
    fig.savefig(out_path, bbox_inches="tight")
    logger.info(f"Figure saved to {out_path}")
    return out_path


def plot_class_balance(real_df, synthetic_dfs: dict, target: str, figures_dir, title_suffix: str = ""):
    """Positive and negative rate of one target across the real data and every synthetic dataset."""
    all_dfs = {"real": real_df} | synthetic_dfs
    methods = list(all_dfs)
    classes = sorted(real_df[target].unique())

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 3, 4.6))
    for i, cls in enumerate(classes):
        freqs = [all_dfs[m][target].value_counts(normalize=True).get(cls, 0.0) for m in methods]
        bars = ax.bar(x + (i - 0.5) * width, freqs, width, label=f"class {cls}", alpha=0.9)
        for bar, freq in zip(bars, freqs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{freq:.0%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Relative frequency")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Class balance of {target}{title_suffix}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    save_figure(fig, f"class_balance_{target}", figures_dir)
    return fig


def plot_marginal_distributions(real_df, synthetic_dfs: dict, col: str, cfg, figures_dir):
    """Distribution of one column in the real data and in each synthetic dataset."""

    is_categorical = col in set(categorical_columns(real_df, cfg))

    all_dfs = {"real": real_df} | synthetic_dfs
    n_cols = min(3, len(all_dfs))
    n_rows = int(np.ceil(len(all_dfs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), sharey=True)
    flat = np.atleast_1d(axes).ravel()

    for ax, (method, df) in zip(flat, all_dfs.items()):
        series = df[col].dropna()
        if is_categorical:
            freq = series.value_counts(normalize=True).sort_index().head(10)
            ax.bar(range(len(freq)), freq.values, color=METHOD_COLORS.get(method, GREY), alpha=0.9)
            ax.set_xticks(range(len(freq)))
            ax.set_xticklabels(freq.index, rotation=45, ha="right", fontsize=10)
        else:
            ax.hist(series, bins=30, color=METHOD_COLORS.get(method, GREY), alpha=0.85, density=True)
        ax.set_title(method, fontsize=12)

    for ax in flat[len(all_dfs):]:
        ax.set_visible(False)
    for ax in flat[::n_cols]:
        ax.set_ylabel("Relative frequency" if is_categorical else "Density")

    fig.suptitle(f"Marginal distribution of {col}")
    fig.tight_layout()

    save_figure(fig, f"marginal_{col}", figures_dir)
    return fig


def plot_fidelity_summary(fidelity_df, figures_dir):
    metrics = [("mmd", "MMD"), ("correlation_mae", "Correlation MAE"), ("mean_kl", "Mean KL divergence")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (col, label) in zip(axes, metrics):
        colours = [METHOD_COLORS.get(m, GREY) for m in fidelity_df["method"]]
        bars = ax.bar(fidelity_df["method"], fidelity_df[col], color=colours, alpha=0.9)
        top = fidelity_df[col].max()
        for bar, value in zip(bars, fidelity_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02,
                    f"{value:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(label)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(label, fontsize=12)
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    fig.suptitle("Fidelity of each synthetic dataset against the real training data")
    fig.tight_layout()

    save_figure(fig, "fidelity_summary", figures_dir)
    return fig


def plot_metric_by_method(metrics_df: pd.DataFrame, metric: str, cfg, figures_dir, title: str = "",
                          reference_line: float | None = None, reference_label: str = ""):
    methods = method_order(metrics_df, cfg)
    classifiers = list(dict.fromkeys(metrics_df["classifier"]))

    x = np.arange(len(classifiers))
    width = 0.8 / len(methods)
    offsets = np.linspace(-(len(methods) - 1) / 2, (len(methods) - 1) / 2, len(methods)) * width

    fig, ax = plt.subplots(figsize=(2.4 * len(classifiers) + 4, 4.8))
    for i, method in enumerate(methods):
        sub = metrics_df[metrics_df["method"] == method].set_index("classifier")
        values = np.array([sub[metric].get(c, np.nan) for c in classifiers], dtype=float)
        ax.bar(x + offsets[i], values, width, label=method,
               color=METHOD_COLORS.get(method, GREY), alpha=0.9, edgecolor="white")
        for position, value in zip(x + offsets[i], values):
            if np.isnan(value):
                ax.text(position, 0, " undefined", rotation=90, ha="center", va="bottom",
                        fontsize=9, color=GREY)

    if reference_line is not None:
        ax.axhline(reference_line, color="grey", ls=":", lw=1.2, label=reference_label)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in classifiers])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"{metric.replace('_', ' ')} by training set")
    ax.legend(title="Trained on", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    save_figure(fig, f"{metric}", figures_dir)
    return fig




LOWER_IS_BETTER = {"brier_score", "mean_dpd", "mean_eod"}
HIGHER_IS_BETTER = {"balanced_accuracy", "f1_macro", "roc_auc", "mean_di"}


def plot_delta_heatmap(delta_df: pd.DataFrame, delta_columns: list, name: str, cfg, figures_dir, title: str = ""):
    """Change from the real baseline per training set and classifier."""

    baseline = cfg["experiments"]["baseline_label"]
    plot_df = delta_df[delta_df["method"] != baseline].copy()
    plot_df["row"] = plot_df["method"] + " / " + plot_df["classifier"].str.replace("_", " ")
    pivot = plot_df.set_index("row")[delta_columns]

    labels = []
    for column in delta_columns:
        metric = column.replace("delta_", "")
        if metric in LOWER_IS_BETTER:
            direction = "lower better"
        elif metric in HIGHER_IS_BETTER:
            direction = "higher better"
        else:
            direction = "1 is parity"
        labels.append(f"{metric.replace('_', ' ')}\n({direction})")
    limit = pivot.abs().to_numpy().max()
    limit = limit if limit > 0 else 1.0

    fig, ax = plt.subplots(figsize=(1.5 * len(delta_columns) + 4, 0.45 * len(pivot) + 2))
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit,
                annot=True, fmt=".3f", annot_kws={"size": 10}, linewidths=0.4,
                linecolor="white", xticklabels=labels, cbar_kws={"shrink": 0.7})
    ax.set_ylabel("")
    ax.set_title(title or "Change from the real baseline")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    save_figure(fig, f"delta_heatmap_{name}", figures_dir)
    return fig


def plot_utility_fairness_scatter(delta_df, utility_col: str, fairness_col: str, cfg, figures_dir):
    """Utility change against fairness change, colour by training set and marker by classifier."""
    baseline = cfg["experiments"]["baseline_label"]
    plot_df = delta_df[delta_df["method"] != baseline]
    classifiers = list(dict.fromkeys(plot_df["classifier"]))
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for _, row in plot_df.iterrows():
        ax.scatter(row[utility_col], row[fairness_col],
                   color=METHOD_COLORS.get(row["method"], GREY),
                   marker=markers[classifiers.index(row["classifier"]) % len(markers)],
                   s=130, edgecolors="black", linewidths=0.4, zorder=3)

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(utility_col.replace("delta_", "change in ").replace("_", " "))
    ax.set_ylabel(fairness_col.replace("delta_", "change in ").replace("_", " "))
    ax.set_title("Utility change against fairness change")

    # Two compact legends instead of one entry per point.
    method_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=METHOD_COLORS.get(m, GREY), label=m)
        for m in method_order(plot_df, cfg)
    ]
    clf_handles = [
        plt.Line2D([], [], marker=markers[i % len(markers)], linestyle="", markersize=9,
                   color="black", label=c.replace("_", " "))
        for i, c in enumerate(classifiers)
    ]
    first = ax.legend(handles=method_handles, title="Trained on",
                      bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.add_artist(first)
    ax.legend(handles=clf_handles, title="Classifier",
              bbox_to_anchor=(1.02, 0.45), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    save_figure(fig, "scatter_utility_vs_fairness", figures_dir)
    return fig


def plot_per_attribute_fairness(fairness_df: pd.DataFrame, metric: str, cfg, figures_dir,
                                title: str = "", name_suffix: str = ""):
    """One fairness metric per protected attribute, red when it fails its threshold."""
    thresholds = cfg["fairness"]
    sub = fairness_df[["attribute", metric]].dropna(subset=[metric])
    sub = sub.sort_values(metric, ascending=(metric != "di"))

    if metric == "di":
        limit = thresholds["di_threshold"]
        fails = sub[metric] < limit
        rule = f"{limit} four fifths rule"
    else:
        limit = thresholds[f"{metric}_threshold"]
        fails = sub[metric] > limit
        rule = f"{limit} concern threshold"

    colours = ["#e74c3c" if f else "#2ecc71" for f in fails]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(sub) + 2))
    ax.barh(sub["attribute"], sub[metric], color=colours, alpha=0.9, edgecolor="white")
    ax.axvline(limit, color="darkorange", lw=1.2, ls="--", label=rule)
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} per protected attribute")
    ax.legend(loc="lower right")
    fig.tight_layout()

    save_figure(fig, f"per_attr_{metric}{name_suffix}", figures_dir)
    return fig


def plot_grouped_bars(pivot: pd.DataFrame, ylabel: str, title: str, figures_dir, name: str,
                      reference_line: float | None = None, reference_label: str = ""):

    horizontal = len(pivot.index) > 6

    if horizontal:
        fig, ax = plt.subplots(figsize=(9, 0.42 * pivot.size + 1.6))
        pivot.iloc[::-1].plot(kind="barh", ax=ax, width=0.8, alpha=0.9, edgecolor="white")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("")
        if reference_line is not None:
            ax.axvline(reference_line, color="grey", ls=":", lw=1.2, label=reference_label)
    else:
        fig, ax = plt.subplots(figsize=(min(13.0, 1.7 * len(pivot.index) + 3.0), 4.8))
        pivot.plot(kind="bar", ax=ax, rot=20, width=0.8, alpha=0.9, edgecolor="white")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        if reference_line is not None:
            ax.axhline(reference_line, color="grey", ls=":", lw=1.2, label=reference_label)

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    save_figure(fig, name, figures_dir)
    return fig
