from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.spatial.distance import cdist

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_fidelity_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    cfg: dict[str, Any],
    method_name: str = "unknown",
) -> dict[str, Any]:
    """Compute column stats, KL divergences, MMD and correlation MAE comparing synthetic to real data."""
    logger.info(f"Computing fidelity report for method={method_name}")

    cat_cols = set(cfg["dataset"].get("categorical_columns", []))
    cat_cols |= set(real_df.select_dtypes(include="object").columns)

    num_cols = [c for c in real_df.columns if c not in cat_cols]
    cat_cols_present = [c for c in cat_cols if c in real_df.columns and c in synthetic_df.columns]
    num_cols_present  = [c for c in num_cols  if c in real_df.columns and c in synthetic_df.columns]

    report: dict[str, Any] = {
        "method":      method_name,
        "n_real":      len(real_df),
        "n_synthetic": len(synthetic_df),
    }

    report["column_stats"] = _column_statistics(
        real_df, synthetic_df, num_cols_present, cat_cols_present
    )

    report["kl_divergences"] = _kl_divergences(real_df, synthetic_df, cat_cols_present)
    mean_kl = (
        np.nanmean(list(report["kl_divergences"].values()))
        if report["kl_divergences"]
        else float("nan")
    )

    report["mmd"] = _mmd_rbf(
        real_df[num_cols_present].dropna(),
        synthetic_df[num_cols_present].dropna(),
    ) if num_cols_present else float("nan")

    report["correlation_mae"] = _correlation_mae(
        real_df[num_cols_present], synthetic_df[num_cols_present]
    ) if len(num_cols_present) >= 2 else float("nan")

    logger.info(
        f"  MMD={report['mmd']:.4f}  corr_MAE={report['correlation_mae']:.4f}  mean_KL={mean_kl:.4f}"
    )
    return report


def pprint_fidelity_report(report: dict[str, Any]) -> None:
    """Pretty-print a fidelity report dict to stdout."""
    print(f"\n{'=' * 65}")
    print(f"  Fidelity Report - method: {report['method']}")
    print(f"{'=' * 65}")
    print(f"  Real rows     : {report['n_real']:,}")
    print(f"  Synthetic rows: {report['n_synthetic']:,}")
    print(f"  MMD (RBF)     : {report['mmd']:.6f}  (lower is better)")
    print(f"  Corr MAE      : {report['correlation_mae']:.4f}  (lower is better)")
    
    kl_divs = report.get("kl_divergences", {})
    if kl_divs:
        mean_kl = np.nanmean(list(kl_divs.values()))
        print(f"  Mean KL div   : {mean_kl:.4f}  (lower is better)")
        print(f"\n  Per-column KL divergences (categorical):")
        for col, kl in sorted(kl_divs.items(), key=lambda x: -x[1]):
            print(f"    {col:<55}  {kl:.4f}")

    print(f"\n  Column statistics (numerical columns - top 10):")
    stats = report.get("column_stats", {}).get("numerical", {})
    for col, s in list(stats.items())[:10]:
        print(
            f"    {col:<45}  "
            f"mean: real={s['real_mean']:.3f} / syn={s['syn_mean']:.3f}  "
            f"std:  real={s['real_std']:.3f} / syn={s['syn_std']:.3f}"
        )
    print(f"{'=' * 65}\n")


def _column_statistics(
    real_df: pd.DataFrame,
    syn_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict[str, Any]:
    """Compute per-column mean/std/median for numerical columns and value frequencies for categoricals."""
    stats: dict[str, Any] = {"numerical": {}, "categorical": {}}

    for col in num_cols:
        r = real_df[col].dropna()
        s = syn_df[col].dropna()
        stats["numerical"][col] = {
            "real_mean":   float(r.mean()),
            "syn_mean":    float(s.mean()),
            "real_std":    float(r.std()),
            "syn_std":     float(s.std()),
            "real_median": float(r.median()),
            "syn_median":  float(s.median()),
        }

    for col in cat_cols:
        r_freq = real_df[col].value_counts(normalize=True, dropna=True).to_dict()
        s_freq = syn_df[col].value_counts(normalize=True, dropna=True).to_dict()
        stats["categorical"][col] = {"real_freq": r_freq, "syn_freq": s_freq}

    return stats


def _kl_divergences(
    real_df: pd.DataFrame,
    syn_df:  pd.DataFrame,
    cat_cols: list[str],
    eps: float = 1e-8,
) -> dict[str, float]:
    """Per categorical column, compute KL(P_real || P_synthetic) with an epsilon to avoid log(0)."""
    kl_scores: dict[str, float] = {}

    for col in cat_cols:
        all_cats = sorted(
            set(real_df[col].dropna().unique()) | set(syn_df[col].dropna().unique())
        )
        r_counts = real_df[col].value_counts()
        s_counts = syn_df[col].value_counts()

        p = np.array([r_counts.get(c, 0) for c in all_cats], dtype=float) + eps
        q = np.array([s_counts.get(c, 0) for c in all_cats], dtype=float) + eps
        p /= p.sum()
        q /= q.sum()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kl = float(np.sum(kl_div(p, q)))
        kl_scores[col] = kl

    return kl_scores


def _mmd_rbf(
    real_arr: pd.DataFrame,
    syn_arr:  pd.DataFrame,
    bandwidth: float | None = None,
    max_samples: int = 1000,
) -> float:
    """RBF-kernel MMD between real and synthetic numerical data (0 = identical), subsampled for speed."""
    rng = np.random.default_rng(42)

    X = real_arr.fillna(0.0).values.astype(float)
    Y = syn_arr.fillna(0.0).values.astype(float)

    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("nan")

    # Median heuristic for the bandwidth when not supplied.
    if bandwidth is None:
        all_data = np.vstack([X, Y])
        dists = cdist(all_data, all_data, "sqeuclidean")
        bandwidth = float(np.median(dists[dists > 0])) / 2.0
        bandwidth = max(bandwidth, 1e-6)

    def rbf_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq_dists = cdist(A, B, "sqeuclidean")
        return np.exp(-sq_dists / (2.0 * bandwidth))

    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)

    mmd2 = (K_xx.mean() - 2.0 * K_xy.mean() + K_yy.mean())
    return float(max(mmd2, 0.0) ** 0.5)  # MMD, not MMD^2


def _correlation_mae(
    real_df: pd.DataFrame,
    syn_df:  pd.DataFrame,
) -> float:
    """Mean absolute error between the Spearman correlation matrices of real and synthetic data."""
    common_cols = [c for c in real_df.columns if c in syn_df.columns]
    if len(common_cols) < 2:
        return float("nan")

    r_corr = real_df[common_cols].corr(method="spearman").values
    s_corr = syn_df[common_cols].corr(method="spearman").values

    mask = np.triu(np.ones_like(r_corr, dtype=bool), k=1)
    diffs = np.abs(r_corr[mask] - s_corr[mask])
    return float(np.nanmean(diffs))