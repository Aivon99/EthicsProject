from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def binarise_attribute(attribute: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Map a protected attribute to 1 for the privileged group and 0 for the rest."""
    spec = cfg["dataset"]["protected_groups"].get(attribute.name)
    if spec is None:
        raise KeyError(f"No group definition for '{attribute.name}' in dataset.protected_groups.")

    if spec.get("split") == "median":
        numeric = pd.to_numeric(attribute, errors="coerce")
        binary = (numeric > numeric.median()).astype("Int64")
        return binary.mask(numeric.isna())

    privileged = set(spec["privileged"])
    binary = attribute.isin(privileged).astype("Int64")
    return binary.mask(attribute.isna())


def compute_fairness_for_attribute(
    y_true,
    y_pred,
    attribute: pd.Series,
    cfg: dict,
) -> dict[str, float]:
    """Compute DPD, EOD, DI and the equalized odds ratio for a single protected attribute."""
    binary = binarise_attribute(attribute, cfg)
    keep = binary.notna().to_numpy()

    group = binary[keep].to_numpy(dtype=int)
    y_true_v = np.asarray(y_true, dtype=int)[keep]
    y_pred_v = np.asarray(y_pred, dtype=int)[keep]

    priv   = group == 1
    unpriv = group == 0

    rate_priv   = _positive_rate(y_pred_v, priv)
    rate_unpriv = _positive_rate(y_pred_v, unpriv)
    tpr_priv    = _tpr(y_true_v, y_pred_v, priv)
    tpr_unpriv  = _tpr(y_true_v, y_pred_v, unpriv)
    fpr_priv    = _fpr(y_true_v, y_pred_v, priv)
    fpr_unpriv  = _fpr(y_true_v, y_pred_v, unpriv)

    results = {
        "n_evaluated": int(keep.sum()),
        "n_missing":   int((~keep).sum()),
        "n_priv":      int(priv.sum()),
        "n_unpriv":    int(unpriv.sum()),
        "rate_priv":   rate_priv,
        "rate_unpriv": rate_unpriv,
        "tpr_priv":    tpr_priv,
        "tpr_unpriv":  tpr_unpriv,
        "fpr_priv":    fpr_priv,
        "fpr_unpriv":  fpr_unpriv,
    }

    # Compute Demographic Parity Difference (DPD)
    results["dpd"] = abs(rate_priv - rate_unpriv)

    # Compute Equalized Odds Difference (EOD)
    delta_tpr = abs(tpr_priv - tpr_unpriv)
    delta_fpr = abs(fpr_priv - fpr_unpriv)
    results["delta_tpr"] = delta_tpr
    results["delta_fpr"] = delta_fpr
    both_missing = np.isnan(delta_tpr) and np.isnan(delta_fpr)
    results["eod"] = float("nan") if both_missing else float(np.nanmax([delta_tpr, delta_fpr]))

    # Compute Disparate Impact (DI)
    if rate_priv == 0 and rate_unpriv == 0:
        results["di"] = float("nan")
    elif rate_priv == 0 or rate_unpriv == 0:
        results["di"] = 0.0
    else:
        results["di"] = float(min(rate_priv, rate_unpriv) / max(rate_priv, rate_unpriv))

    # Equalized-odds odds-ratio (Marrero et al., ECAI 2024): recall of the
    # unprivileged group relative to the privileged (reference) group.
    #   ~1.0 -> fair (equal recall); <1.0 -> favours privileged group;
    #   >1.0 -> favours unprivileged group.
    support_priv = int((priv & (y_true_v == 1)).sum())
    support_unpriv = int((unpriv & (y_true_v == 1)).sum())
    results["support_priv_pos"] = support_priv
    results["support_unpriv_pos"] = support_unpriv

    min_support = cfg["fairness"]["min_odds_ratio_support"]
    too_thin = support_priv < min_support or support_unpriv < min_support
    if too_thin or np.isnan(tpr_priv) or np.isnan(tpr_unpriv) or tpr_priv == 0:
        results["odds_ratio"] = float("nan")
    else:
        results["odds_ratio"] = float(tpr_unpriv / tpr_priv)

    return results


def compute_all_fairness_metrics(
    y_true,
    y_pred,
    test_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Compute fairness metrics for every configured protected attribute, one row each."""
    subset = cfg.get("fairness_attributes_subset")
    attributes = subset if subset is not None else cfg["dataset"]["protected_attributes"]

    rows = []
    for attr in attributes:
        if attr not in test_df.columns:
            logger.warning(f"Protected attribute '{attr}' not found in the test data, skipping.")
            continue
        metrics = compute_fairness_for_attribute(y_true, y_pred, test_df[attr], cfg)
        metrics["attribute"] = attr
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["attribute"] + [c for c in df.columns if c != "attribute"]]
    return df


def summarise_fairness(fairness_df: pd.DataFrame) -> dict[str, float]:
    return {
        f"mean_{metric}": float(fairness_df[metric].mean(skipna=True))
        for metric in ("dpd", "eod", "di", "odds_ratio")
        if metric in fairness_df.columns
    }


def _positive_rate(y_pred, mask) -> float:
    group = y_pred[mask]
    return float("nan") if len(group) == 0 else float(group.mean())


def _tpr(y_true, y_pred, mask) -> float:
    pos = mask & (y_true == 1)
    return float("nan") if pos.sum() == 0 else float(y_pred[pos].mean())


def _fpr(y_true, y_pred, mask) -> float:
    neg = mask & (y_true == 0)
    return float("nan") if neg.sum() == 0 else float(y_pred[neg].mean())
