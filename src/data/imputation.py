from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def fit_imputation(X_train: pd.DataFrame, sensitive_cols: list) -> dict[str, dict]:
    """Learn one fill value per column from the training split only, so nothing leaks from the test split.

    Protected attributes get no fill value at all. They are flagged instead, because imputing a
    circumstance would invent group membership and bias the fairness metrics that follow.
    """
    plan: dict[str, dict] = {}

    for col in X_train.columns:
        if X_train[col].isna().sum() == 0:
            continue

        if col in sensitive_cols:
            plan[col] = {"strategy": "flag"}
        elif _is_categorical(X_train[col]):
            plan[col] = {"strategy": "unknown", "value": "UNKNOWN"}
        elif _is_boolean(X_train[col]) or _is_ordinal_integer(X_train[col]):
            mode = X_train[col].mode(dropna=True)
            if len(mode):
                plan[col] = {"strategy": "mode", "value": mode[0]}
        else:
            plan[col] = {"strategy": "median", "value": X_train[col].median(skipna=True)}

    n_flagged = sum(1 for p in plan.values() if p["strategy"] == "flag")
    logger.info(f"Imputation plan fitted on the training split: {len(plan) - n_flagged} columns "
                f"filled, {n_flagged} protected attributes flagged instead")
    return plan


def apply_imputation(X: pd.DataFrame, plan: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fitted plan, returning the filled frame and the missingness flags for protected attributes."""
    filled = X.copy()
    flags = {}

    for col, spec in plan.items():
        if col not in filled.columns:
            continue
        if spec["strategy"] == "flag":
            flags[f"{col}_was_nan"] = filled[col].isna()
        else:
            filled[col] = filled[col].fillna(spec["value"])

    return filled, pd.DataFrame(flags, index=X.index)


def imputation_report(X_before: pd.DataFrame, X_after: pd.DataFrame, plan: dict[str, dict]) -> pd.DataFrame:
    """Per column comparison of missingness before and after, with the strategy that was applied."""
    before = X_before.isna().sum()
    rows = []
    for col, spec in plan.items():
        rows.append(
            {
                "column": col,
                "missing_before": int(before[col]),
                "missing_rate_before": float(before[col] / len(X_before)),
                "missing_after": int(X_after[col].isna().sum()),
                "strategy": spec["strategy"],
            }
        )
    return pd.DataFrame(rows).sort_values("missing_before", ascending=False).reset_index(drop=True)


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype) == "category"


def _is_boolean(series: pd.Series) -> bool:
    values = series.dropna().unique()
    return len(values) > 0 and set(values).issubset({True, False, 0, 1})


def _is_ordinal_integer(series: pd.Series) -> bool:
    """Likert style: a small number of distinct values on a short integer scale."""
    if series.dtype not in [np.int64, np.float64, "Int64"]:
        return False
    return series.nunique(dropna=True) <= 10 and series.max(skipna=True) <= 10

