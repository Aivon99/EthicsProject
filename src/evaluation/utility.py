from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_utility_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Compute balanced accuracy, F1-macro, ROC-AUC and Brier score for one set of test predictions."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    results: dict[str, float] = {}

    results["balanced_accuracy"] = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    results["f1_macro"] = float(
        f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    )
    try:
        results["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        results["roc_auc"] = float("nan")
    results["brier_score"] = float(brier_score_loss(y_true_arr, y_prob_arr))

    return results


def compute_mmd(
    X_real: np.ndarray | pd.DataFrame,
    X_synthetic: np.ndarray | pd.DataFrame,
    cfg: dict[str, Any],
) -> float:
    """Estimate MMD^2 (RBF kernel, unbiased, subsampled) between real and synthetic features."""
    mmd_cfg = cfg["mmd"]
    n_sub   = mmd_cfg["n_subsample"]
    gamma   = mmd_cfg["gamma"]

    rng = np.random.default_rng(cfg["seed"])

    X = np.asarray(X_real,      dtype=float)
    Y = np.asarray(X_synthetic, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    Y = Y[~np.isnan(Y).any(axis=1)]

    if n_sub is not None and len(X) > n_sub:
        X = X[rng.choice(len(X), size=n_sub, replace=False)]
    if n_sub is not None and len(Y) > n_sub:
        Y = Y[rng.choice(len(Y), size=n_sub, replace=False)]

    if gamma is None:
        combined  = np.vstack([X[:500], Y[:500]])
        sub       = combined[rng.choice(len(combined), size=min(500, len(combined)), replace=False)]
        sq_d      = np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1)
        median_sq = np.median(sq_d[sq_d > 0])
        gamma     = 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0

    K_XX = _rbf(X, X, gamma)
    K_YY = _rbf(Y, Y, gamma)
    K_XY = _rbf(X, Y, gamma)
    n, m = len(X), len(Y)

    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    mmd2 = (
        K_XX.sum() / (n * (n - 1))
        + K_YY.sum() / (m * (m - 1))
        - 2.0 * K_XY.mean()
    )
    return float(max(mmd2, 0.0))


def column_correlation_delta(
    X_real: pd.DataFrame,
    X_synth: pd.DataFrame,
) -> dict[str, float | dict[str, float]]:
    """Return per-pair absolute Pearson correlation deltas plus their mean and max."""
    num_cols   = X_real.select_dtypes(include="number").columns.tolist()
    corr_real  = X_real[num_cols].corr()
    corr_synth = X_synth[num_cols].corr()
    delta      = (corr_real - corr_synth).abs()
    mask       = np.triu(np.ones(delta.shape, dtype=bool), k=1)

    pairs: dict[str, float] = {}
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if mask[i, j]:
                pairs[f"{c1}|{c2}"] = float(delta.loc[c1, c2])

    values = list(pairs.values())
    return {
        "mean_abs_delta": float(np.mean(values)) if values else 0.0,
        "max_abs_delta":  float(np.max(values))  if values else 0.0,
        "per_pair":       pairs,
    }


def utility_delta(
    real_metrics: dict[str, float],
    synth_metrics: dict[str, float],
) -> dict[str, float]:
    """Return delta (real vs. synthetic) for the three core utility metrics."""
    keys = ["balanced_accuracy", "f1_macro", "roc_auc"]
    return {f"delta_{k}": real_metrics[k] - synth_metrics[k] for k in keys}


def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    sq_norm = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sq_norm)
