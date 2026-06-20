from __future__ import annotations

from typing import Any

import pandas as pd



def compute_delta_matrix(
    metrics_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute per-row deltas relative to the real-data baseline.

    For each (method, classifier) pair the delta is the value minus the
    corresponding value of the same classifier trained on real data.

    Parameters
    ----------
    metrics_df:
        Output of :func:`run_full_experiment_matrix` — one row per
        (method, classifier).
    cfg:
        Parsed configuration dict (used to identify the baseline label).

    Returns
    -------
    pd.DataFrame with the same index and ``method``/``classifier`` columns
    as *metrics_df* plus additional ``delta_<metric>`` columns. The baseline
    rows have delta = 0 by definition.
    """
    baseline_label  = cfg["experiments"]["baseline_label"]
    numeric_cols    = metrics_df.select_dtypes(include="number").columns.tolist()
    delta_df        = metrics_df.copy()

    # Build a lookup: classifier → baseline row.
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
    fair_prefixes = ("delta_mean_dpd", "delta_mean_eod", "delta_mean_di")
    return [c for c in metrics_df.columns if c.startswith(fair_prefixes)]
