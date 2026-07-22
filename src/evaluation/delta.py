from __future__ import annotations

from typing import Any

import pandas as pd



def compute_delta_matrix(
    metrics_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Add delta_<metric> columns giving each row's change from the same classifier's real baseline."""
    baseline_label  = cfg["experiments"]["baseline_label"]
    numeric_cols    = metrics_df.select_dtypes(include="number").columns.tolist()
    delta_df        = metrics_df.copy()

    # Lookup: classifier -> its baseline row.
    baseline_rows = (
        metrics_df[metrics_df["method"] == baseline_label]
        .set_index("classifier")[numeric_cols]
    )

    for col in numeric_cols:
        delta_col = f"delta_{col}"
        delta_df[delta_col] = delta_df.apply(
            lambda row: (
                row[col] - baseline_rows.loc[row["classifier"], col]
                if row["classifier"] in baseline_rows.index
                else float("nan")
            ),
            axis=1,
        )

    return delta_df


def get_utility_delta_columns(metrics_df: pd.DataFrame) -> list[str]:
    """Return a list of delta column names that correspond to utility metrics."""
    util_prefixes = ("delta_balanced_accuracy", "delta_f1_macro",
                     "delta_roc_auc", "delta_brier_score", "delta_mmd")
    return [c for c in metrics_df.columns if c.startswith(util_prefixes)]


def get_fairness_delta_columns(metrics_df: pd.DataFrame) -> list[str]:
    """Return a list of delta column names that correspond to fairness metrics."""
    fair_prefixes = ("delta_mean_dpd", "delta_mean_eod", "delta_mean_di", "delta_mean_odds_ratio")
    return [c for c in metrics_df.columns if c.startswith(fair_prefixes)]
