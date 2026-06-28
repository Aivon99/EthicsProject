import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype) == "category"


def _is_boolean(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return set(non_null.unique()).issubset({True, False, 0, 1})


def _is_ordinal_integer(series: pd.Series) -> bool:
    """
    Heuristic: integer dtype with few unique values => treat as ordinal/Likert.
    We use <=10 unique values as the cutoff.
    """
    if series.dtype not in [np.int64, np.float64, "Int64"]:
        return False
    n_unique = series.nunique(dropna=True)
    col_min  = series.min(skipna=True)
    col_max  = series.max(skipna=True)
    # Likert-like: small range, small number of unique values
    return n_unique <= 10 and col_max <= 10


def impute(
    X: pd.DataFrame,
    sensitive_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in X.

    Rules
    -----
    - Sensitive cols      : never imputed; a boolean missingness flag col is added instead
    - Categorical (object): NaN → "UNKNOWN"
    - Boolean             : NaN → mode
    - Ordinal integer     : NaN → mode  (Likert scales, small-range integers)
    - Continuous float    : NaN → median

    Parameters
    ----------
    X               : feature dataframe (output of preprocess())
    sensitive_cols  : columns to flag but not impute

    Returns
    -------
    X_imputed   : dataframe with NaNs filled
    flag_df     : boolean dataframe of which sensitive values were originally NaN
                  (same index as X_imputed, one col per sensitive col that had NaNs)
    """
    df = X.copy()
    flag_cols = {}

    n_cols     = df.shape[1]
    n_imputed  = 0
    n_flagged  = 0

    for col in df.columns:
        n_nan = df[col].isna().sum()
        if n_nan == 0:
            continue

        # ── Sensitive: flag only, never impute ───────────────────────────────
        if col in sensitive_cols:
            flag_cols[f"{col}_was_nan"] = df[col].isna()
            n_flagged += 1
            logger.info(f"  [FLAG]    {col:<45} {n_nan} NaNs ({n_nan/len(df):.1%})")
            continue

        # ── Categorical ───────────────────────────────────────────────────────
        if _is_categorical(df[col]):
            df[col] = df[col].fillna("UNKNOWN")
            n_imputed += 1
            logger.info(f"  [CAT]     {col:<45} {n_nan} NaNs → 'UNKNOWN'")
            continue

        # ── Boolean ───────────────────────────────────────────────────────────
        if _is_boolean(df[col]):
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                n_imputed += 1
                logger.info(f"  [BOOL]    {col:<45} {n_nan} NaNs → mode={mode_val[0]}")
            continue

        # ── Ordinal integer ───────────────────────────────────────────────────
        if _is_ordinal_integer(df[col]):
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                n_imputed += 1
                logger.info(f"  [ORD]     {col:<45} {n_nan} NaNs → mode={mode_val[0]:.0f}")
            continue

        # ── Continuous ────────────────────────────────────────────────────────
        median_val = df[col].median(skipna=True)
        df[col] = df[col].fillna(median_val)
        n_imputed += 1
        logger.info(f"  [CONT]    {col:<45} {n_nan} NaNs → median={median_val:.3f}")

    flag_df = pd.DataFrame(flag_cols, index=df.index)

    logger.info(
        f"\nImputation done: {n_imputed}/{n_cols} cols imputed, "
        f"{n_flagged} sensitive cols flagged (not imputed)"
    )
    remaining = df.isna().sum().sum()
    if remaining > 0:
        logger.warning(f"  {remaining} NaNs still remain (in sensitive cols — expected)")

    return df, flag_df


def imputation_report(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    flag_df: pd.DataFrame,
    sensitive_cols: list[str],
) -> pd.DataFrame:
    """
    Returns a tidy summary dataframe comparing NaN counts before and after imputation.
    Useful for notebook display.
    """
    before = X_before.isna().sum().rename("nan_before")
    after  = X_after.isna().sum().rename("nan_after")
    report = pd.concat([before, after], axis=1)
    report["nan_fraction_before"] = before / len(X_before)
    report["nan_fraction_after"]  = after  / len(X_after)
    report["strategy"] = "none"

    for col in report.index:
        if col in sensitive_cols:
            report.loc[col, "strategy"] = "flagged"
        elif before[col] == 0:
            report.loc[col, "strategy"] = "no_nan"
        elif _is_categorical(X_before[col]):
            report.loc[col, "strategy"] = "cat→UNKNOWN"
        elif _is_boolean(X_before[col]):
            report.loc[col, "strategy"] = "bool→mode"
        elif _is_ordinal_integer(X_before[col]):
            report.loc[col, "strategy"] = "ord→mode"
        else:
            report.loc[col, "strategy"] = "cont→median"

    return report.query("nan_before > 0").sort_values("nan_before", ascending=False)