from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from src.utils import get_logger





logger = get_logger(__name__)

def compute_clf_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def maximum_mean_discrepancy(
    X_real: pd.DataFrame | np.ndarray,
    X_synth: pd.DataFrame | np.ndarray,
    *,
    gamma: float = 1.0,
) -> float:
    """Unbiased MMD² with RBF kernel."""
    if isinstance(X_real, pd.DataFrame):
        X_real = X_real.values
    if isinstance(X_synth, pd.DataFrame):
        X_synth = X_synth.values

    X_real = X_real.astype(float)
    X_synth = X_synth.astype(float)

    def rbf(A, B):
        diff = A[:, None, :] - B[None, :, :]
        return np.exp(-gamma * np.sum(diff ** 2, axis=-1))

    K_rr = rbf(X_real, X_real)
    K_ss = rbf(X_synth, X_synth)
    K_rs = rbf(X_real, X_synth)
    n, m = len(X_real), len(X_synth)
    np.fill_diagonal(K_rr, 0.0)
    np.fill_diagonal(K_ss, 0.0)
    return float(
        K_rr.sum() / (n * (n - 1))
        + K_ss.sum() / (m * (m - 1))
        - 2 * K_rs.mean()
    )


def column_correlation_delta(X_real: pd.DataFrame, X_synth: pd.DataFrame) -> dict[str, float]:
    num_cols = X_real.select_dtypes(include="number").columns.tolist()
    corr_real = X_real[num_cols].corr()
    corr_synth = X_synth[num_cols].corr()
    delta = (corr_real - corr_synth).abs()
    mask = np.triu(np.ones(delta.shape, dtype=bool), k=1)
    pairs = {}
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if mask[i, j]:
                pairs[f"{c1}|{c2}"] = float(delta.loc[c1, c2])
    values = list(pairs.values())
    return {
        "mean_abs_delta": float(np.mean(values)) if values else 0.0,
        "max_abs_delta":  float(np.max(values)) if values else 0.0,
        "per_pair": pairs,
    }


def utility_delta(real_metrics: dict, synth_metrics: dict) -> dict[str, float]:
    keys = ["balanced_accuracy", "f1_macro", "roc_auc"]
    return {f"delta_{k}": real_metrics[k] - synth_metrics[k] for k in keys}