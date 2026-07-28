from __future__ import annotations

from typing import Any

import pandas as pd


UTILITY_METRICS = ["balanced_accuracy", "f1_macro", "roc_auc", "brier_score"]
FAIRNESS_METRICS = ["mean_dpd", "mean_eod", "mean_di", "mean_odds_ratio"]


def compute_delta_matrix(metrics_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add a delta column per metric, measured as synthetic minus the same classifier's real baseline.

    A negative delta always means the synthetic source is worse on a higher-is-better metric.
    """
    baseline_label = cfg["experiments"]["baseline_label"]
    delta_df = metrics_df.copy()

    baseline = metrics_df[metrics_df["method"] == baseline_label].set_index("classifier")
    metrics = [m for m in UTILITY_METRICS + FAIRNESS_METRICS if m in metrics_df.columns]

    for metric in metrics:
        reference = delta_df["classifier"].map(baseline[metric])
        delta_df[f"delta_{metric}"] = delta_df[metric] - reference
        delta_df.loc[delta_df["method"] == baseline_label, f"delta_{metric}"] = float("nan")

    return delta_df


def get_utility_delta_columns(metrics_df: pd.DataFrame) -> list[str]:
    return [f"delta_{m}" for m in UTILITY_METRICS if f"delta_{m}" in metrics_df.columns]


def get_fairness_delta_columns(metrics_df: pd.DataFrame) -> list[str]:
    return [f"delta_{m}" for m in FAIRNESS_METRICS if f"delta_{m}" in metrics_df.columns]
