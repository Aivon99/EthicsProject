from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def _check_binary_sensitive(sensitive: pd.Series):
    vals = sorted(sensitive.unique())
    if len(vals) != 2:
        raise ValueError(f"Sensitive attribute must be binary; got values: {vals}.")
    return vals[0], vals[1]


def _positive_rate(y_pred, mask) -> float:
    group = y_pred[mask]
    return float("nan") if len(group) == 0 else float(group.mean())


def _tpr(y_true, y_pred, mask) -> float:
    pos_mask = mask & (y_true == 1)
    return float("nan") if pos_mask.sum() == 0 else float(y_pred[pos_mask].mean())


def _fpr(y_true, y_pred, mask) -> float:
    neg_mask = mask & (y_true == 0)
    return float("nan") if neg_mask.sum() == 0 else float(y_pred[neg_mask].mean())


def demographic_parity_difference(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    """DPD = |Pr[Ŷ=1|A=0] - Pr[Ŷ=1|A=1]|"""
    unpriv, priv = _check_binary_sensitive(sensitive)
    return abs(
        _positive_rate(y_pred, (sensitive == unpriv).values)
        - _positive_rate(y_pred, (sensitive == priv).values)
    )


def equalized_odds_difference(y_true, y_pred, sensitive: pd.Series) -> float:
    """EOD = max(|ΔTPR|, |ΔFPR|)"""
    unpriv, priv = _check_binary_sensitive(sensitive)
    m0 = (sensitive == unpriv).values
    m1 = (sensitive == priv).values
    tpr_gap = abs(_tpr(y_true, y_pred, m0) - _tpr(y_true, y_pred, m1))
    fpr_gap = abs(_fpr(y_true, y_pred, m0) - _fpr(y_true, y_pred, m1))
    return max(tpr_gap, fpr_gap)


def disparate_impact(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    """DI = Pr[Ŷ=1|A=0] / Pr[Ŷ=1|A=1]"""
    unpriv, priv = _check_binary_sensitive(sensitive)
    pr_unpriv = _positive_rate(y_pred, (sensitive == unpriv).values)
    pr_priv   = _positive_rate(y_pred, (sensitive == priv).values)
    if pr_priv == 0:
        logger.warning("Privileged group positive rate is 0; returning NaN for DI.")
        return float("nan")
    return pr_unpriv / pr_priv


def compute_fairness_metrics(y_true, y_pred, sensitive: pd.Series, attr_name: str = "sensitive") -> dict:
    return {
        f"{attr_name}_dpd": demographic_parity_difference(y_pred, sensitive),
        f"{attr_name}_eod": equalized_odds_difference(y_true, y_pred, sensitive),
        f"{attr_name}_di":  disparate_impact(y_pred, sensitive),
    }


def fairness_delta(real_fairness: dict, synth_fairness: dict) -> dict[str, float]:
    return {k: abs(synth_fairness[k] - real_fairness[k]) for k in real_fairness}


