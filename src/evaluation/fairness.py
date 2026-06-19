from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)



# ---- Public functions --------------------------------------------------------
def compute_fairness_for_attribute(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    attribute: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute DPD, EOD, and DI for a single protected attribute.

    Multi-category attributes are binarised via :func:`_binarise_attribute`
    before metric computation. The set of metrics computed is controlled by
    ``cfg["fairness_metrics"]``.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (real test set).
    y_pred:
        Hard predictions (0/1) from the trained model.
    attribute:
        Raw values of the protected attribute from the test set.
    cfg:
        Parsed configuration dict.

    Returns
    -------
    dict with keys ``"dpd"``, ``"eod"``, ``"di"`` (depending on config) plus
    auxiliary diagnostics ``"rate_priv"``, ``"rate_unpriv"``, ``"tpr_priv"``,
    ``"tpr_unpriv"``, ``"fpr_priv"``, ``"fpr_unpriv"``.
    """
    y_true_arr  = np.asarray(y_true, dtype=int)
    y_pred_arr  = np.asarray(y_pred, dtype=int)
    binary_attr = _binarise_attribute(attribute)

    valid_mask  = ~binary_attr.isna()
    binary_vals = binary_attr[valid_mask].values.astype(int)
    idx         = valid_mask.values
    y_true_v    = y_true_arr[idx]
    y_pred_v    = y_pred_arr[idx]

    priv_mask   = binary_vals == 1
    unpriv_mask = binary_vals == 0

    rate_priv   = _positive_rate(y_pred_v, priv_mask)
    rate_unpriv = _positive_rate(y_pred_v, unpriv_mask)
    tpr_priv    = _tpr(y_true_v, y_pred_v, priv_mask)
    tpr_unpriv  = _tpr(y_true_v, y_pred_v, unpriv_mask)
    fpr_priv    = _fpr(y_true_v, y_pred_v, priv_mask)
    fpr_unpriv  = _fpr(y_true_v, y_pred_v, unpriv_mask)

    results: dict[str, float] = {
        "rate_priv":   rate_priv,
        "rate_unpriv": rate_unpriv,
        "tpr_priv":    tpr_priv,
        "tpr_unpriv":  tpr_unpriv,
        "fpr_priv":    fpr_priv,
        "fpr_unpriv":  fpr_unpriv,
    }

    # Compute Demographic Parity Difference (DPD)
    #   DPD = |Pr[Ŷ=1|A=0] - Pr[Ŷ=1|A=1]|
    results["dpd"] = abs(rate_priv - rate_unpriv)


    # Compute Equalized Odds Difference (EOD)
    #   EOD = max(|ΔTPR|, |ΔFPR|)
    delta_tpr = abs(tpr_priv - tpr_unpriv) if not (np.isnan(tpr_priv) or np.isnan(tpr_unpriv)) else float("nan")
    delta_fpr = abs(fpr_priv - fpr_unpriv) if not (np.isnan(fpr_priv) or np.isnan(fpr_unpriv)) else float("nan")
    
    results["delta_tpr"] = delta_tpr
    results["delta_fpr"] = delta_fpr
    results["eod"]       = float(np.nanmax([delta_tpr, delta_fpr]))

    # Compute Disparate Impact (DI)
    #   DI = Pr[Ŷ=1|A=0] / Pr[Ŷ=1|A=1]
    if rate_priv == 0 and rate_unpriv == 0:
        results["di"] = float("nan")
    elif rate_priv == 0 or rate_unpriv == 0:
        results["di"] = 0.0
    else:
        results["di"] = float(min(rate_priv, rate_unpriv) / max(rate_priv, rate_unpriv))

    return results


def compute_all_fairness_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute fairness metrics for every configured protected attribute.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (real test set).
    y_pred:
        Hard predictions (0/1) from the trained model.
    test_df:
        The full real test DataFrame used to extract attribute columns.
    cfg:
        Parsed configuration dict.

    Returns
    -------
    pd.DataFrame with one row per protected attribute and columns for each
    enabled fairness metric plus auxiliary diagnostics.
    """
    subset     = cfg.get("fairness_attributes_subset")
    all_attrs  = cfg["dataset"]["protected_attributes"]
    attributes = subset if subset is not None else all_attrs

    rows = []
    for attr in attributes:
        if attr not in test_df.columns:
            logger.warning("Protected attribute '%s' not found in test_df. Skipping.", attr)
            continue
        metrics              = compute_fairness_for_attribute(y_true, y_pred, test_df[attr], cfg)
        metrics["attribute"] = attr
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["attribute"] + [c for c in df.columns if c != "attribute"]
        df   = df[cols]
    return df


def summarise_fairness(fairness_df: pd.DataFrame) -> dict[str, float]:
    """Return mean DPD, EOD, and DI across all protected attributes."""
    summary: dict[str, float] = {}
    for metric in ("dpd", "eod", "di"):
        if metric in fairness_df.columns:
            summary[f"mean_{metric}"] = float(fairness_df[metric].mean(skipna=True))
    return summary



# ---- Internal helpers --------------------------------------------------------
def _positive_rate(y_pred, mask) -> float:
    group = y_pred[mask]
    return float("nan") if len(group) == 0 else float(group.mean())


def _tpr(y_true, y_pred, mask) -> float:
    pos_mask = mask & (y_true == 1)
    return float("nan") if pos_mask.sum() == 0 else float(y_pred[pos_mask].mean())


def _fpr(y_true, y_pred, mask) -> float:
    neg_mask = mask & (y_true == 0)
    return float("nan") if neg_mask.sum() == 0 else float(y_pred[neg_mask].mean())


def _binarise_attribute(series: pd.Series) -> pd.Series:
    """Convert a protected attribute to binary 0/1 via median split.

    Already-binary columns (two unique values) are mapped to 0/1 directly.
    Ordinal or continuous columns are split at the median; categorical
    strings are ordered alphabetically and split at the median code.

    Parameters
    ----------
    series:
        Raw protected attribute column.

    Returns
    -------
    pd.Series of dtype Int64 with values in {0, 1}.
    """
    unique_vals = series.dropna().unique()

    if len(unique_vals) <= 2:
        s_sorted = sorted(unique_vals)
        mapping  = {s_sorted[0]: 0, s_sorted[-1]: 1}
        return series.map(mapping).astype("Int64")

    try:
        numeric = pd.to_numeric(series, errors="raise")
        return (numeric > numeric.median()).astype(int)
    except (ValueError, TypeError):
        codes  = series.astype("category").cat.codes
        median = codes[codes >= 0].median()
        return (codes > median).astype("Int64")
