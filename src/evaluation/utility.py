from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)



# ---- Public functions --------------------------------------------------------

def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute the core predictive utility metrics used in the proposal."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro":          float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
    }

    if y_score is not None:
        y_score_arr = np.asarray(y_score)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def tstr_gap(
    real_train_metrics: Mapping[str, float],
    synthetic_train_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Compute metric-wise real-vs-synthetic train gaps on held-out real test data."""
    shared_metrics = sorted(set(real_train_metrics).intersection(synthetic_train_metrics))
    return {
        metric: float(real_train_metrics[metric] - synthetic_train_metrics[metric])
        for metric in shared_metrics
    }


def compute_utility_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Compute enabled classifier utility metrics for a single (model, test-set) pair.

    MMD is excluded here => use :func:`compute_mmd` for distributional fidelity.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels from the real test set.
    y_pred:
        Hard predictions (0/1) from the trained classifier.
    y_prob:
        Probability estimates for the positive class (shape ``(n,)``).
    cfg:
        Parsed configuration dict.

    Returns
    -------
    dict mapping metric name -> scalar value
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    results: dict[str, float] = {}

    # Compute balanced accuracy
    results["balanced_accuracy"] = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    # Compute f1 macro score
    results["f1_macro"] = float(
        f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    )

    # Compute ROC AUC
    try:
        results["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        results["roc_auc"] = float("nan")

    # Compute Brier score
    results["brier_score"] = float(brier_score_loss(y_true_arr, y_prob_arr))

    return results


def compute_mmd(
    X_real: np.ndarray | pd.DataFrame,
    X_synthetic: np.ndarray | pd.DataFrame,
    cfg: dict[str, Any],
) -> float:
    """Estimate MMD^2 between real and synthetic features.

    Uses an unbiased two-sample estimator with an RBF kernel and supports
    subsampling for large datasets. Bandwidth is set via the median heuristic
    when ``cfg["mmd"]["gamma"]`` is *null*.

    Parameters
    ----------
    X_real:
        Feature matrix from the real training set.
    X_synthetic:
        Feature matrix from the synthetic training set.
    cfg:
        Parsed configuration dict. Reads ``cfg["mmd"]`` for kernel settings
        and ``cfg["seed"]`` for reproducible subsampling.

    Returns
    -------
    float MMD^2 estimate (>= 0, where 0 means identical distributions).
    """
    mmd_cfg = cfg["mmd"]
    n_sub   = mmd_cfg["n_subsample"]
    gamma   = mmd_cfg["gamma"]

    # Random NumPy Generator
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


def correlation_preservation(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "spearman",
) -> dict[str, float]:
    """Measure how much the correlation structure changes from real to synthetic data.

    Parameters
    ----------
    real_df:
        Feature DataFrame from the real dataset.
    synthetic_df:
        Feature DataFrame from the synthetic dataset.
    method:
        Correlation method passed to :meth:`pandas.DataFrame.corr`
        (``"pearson"``, ``"spearman"``, or ``"kendall"``).

    Returns
    -------
    dict with keys ``"corr_abs_diff_mean"`` and ``"corr_abs_diff_max"``.
    """
    shared_cols = [c for c in real_df.columns if c in synthetic_df.columns]
    if not shared_cols:
        raise ValueError("real_df and synthetic_df do not share any columns")

    real_corr  = real_df[shared_cols].corr(method=method)
    synth_corr = synthetic_df[shared_cols].corr(method=method)
    diff       = (real_corr - synth_corr).abs().values

    return {
        "corr_abs_diff_mean": float(np.nanmean(diff)),
        "corr_abs_diff_max":  float(np.nanmax(diff)),
    }


def column_correlation_delta(
    X_real: pd.DataFrame,
    X_synth: pd.DataFrame,
) -> dict[str, float | dict[str, float]]:
    """Return per-pair Pearson correlation deltas and aggregate summary stats.

    Parameters
    ----------
    X_real:
        Feature DataFrame from the real dataset.
    X_synth:
        Feature DataFrame from the synthetic dataset.

    Returns
    -------
    dict with keys ``"mean_abs_delta"``, ``"max_abs_delta"``, and ``"per_pair"``
    (a nested dict mapping ``"col1|col2"`` -> absolute delta).
    """
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



# ---- Internal helpers --------------------------------------------------------

def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    sq_norm = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sq_norm)
