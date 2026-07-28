from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_utility_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Compute balanced accuracy, F1-macro, ROC-AUC and Brier score for one set of test predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    results = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
    
    results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return results


def compute_mmd(X_real, X_synthetic, cfg: dict[str, Any]) -> float:
    """Estimate the squared Maximum Mean Discrepancy (MMD) between real and synthetic features."""
    n_subsample = cfg["mmd"]["n_subsample"]
    gamma = cfg["mmd"]["gamma"]
    rng = np.random.default_rng(cfg["seed"])

    X = np.asarray(X_real, dtype=float)
    Y = np.asarray(X_synthetic, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    Y = Y[~np.isnan(Y).any(axis=1)]

    if n_subsample is not None and len(X) > n_subsample:
        X = X[rng.choice(len(X), size=n_subsample, replace=False)]
    if n_subsample is not None and len(Y) > n_subsample:
        Y = Y[rng.choice(len(Y), size=n_subsample, replace=False)]

    if gamma is None:
        gamma = _median_heuristic(X, Y, rng)

    K_XX = _rbf(X, X, gamma)
    K_YY = _rbf(Y, Y, gamma)
    K_XY = _rbf(X, Y, gamma)
    n, m = len(X), len(Y)

    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    mmd2 = K_XX.sum() / (n * (n - 1)) + K_YY.sum() / (m * (m - 1)) - 2.0 * K_XY.mean()
    return float(max(mmd2, 0.0))


def column_correlation_delta(X_real: pd.DataFrame, X_synth: pd.DataFrame) -> dict[str, float]:
    """Return the mean and max absolute change in pairwise correlation between real and synthetic."""
    num_cols = X_real.select_dtypes(include="number").columns.tolist()
    delta = (X_real[num_cols].corr() - X_synth[num_cols].corr()).abs()
    upper = delta.where(np.triu(np.ones(delta.shape, dtype=bool), k=1))
    values = upper.stack().to_numpy()

    return {
        "mean_abs_delta": float(np.nanmean(values)) if values.size else 0.0,
        "max_abs_delta": float(np.nanmax(values)) if values.size else 0.0,
    }


def _median_heuristic(X: np.ndarray, Y: np.ndarray, rng) -> float:
    combined = np.vstack([X[:500], Y[:500]])
    sample = combined[rng.choice(len(combined), size=min(500, len(combined)), replace=False)]
    sq_dist = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
    median_sq = np.median(sq_dist[sq_dist > 0])
    return 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0


def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """Radial basis function kernel."""
    sq_norm = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sq_norm)
