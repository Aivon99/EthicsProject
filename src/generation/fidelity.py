from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import kl_div

from src.evaluation.utility import compute_mmd
from src.generation.metadata import categorical_columns
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_fidelity_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    cfg: dict,
    method_name: str,
) -> dict[str, Any]:
    """Compute column stats, KL divergences, MMD and correlation MAE comparing synthetic to real data."""

    logger.info(f"Fidelity report for {method_name}")

    cat_cols = set(categorical_columns(real_df, cfg))
    cat_present = [c for c in real_df.columns if c in cat_cols and c in synthetic_df.columns]
    num_present = [
        c for c in real_df.columns
        if c not in cat_cols and c in synthetic_df.columns
    ]

    report = {
        "method": method_name,
        "n_real": len(real_df),
        "n_synthetic": len(synthetic_df),
        "kl_divergences": _kl_divergences(real_df, synthetic_df, cat_present),
        "column_stats": _numerical_statistics(real_df, synthetic_df, num_present),
        "mmd": compute_mmd(real_df[num_present].dropna(), synthetic_df[num_present].dropna(), cfg),
        "correlation_mae": _correlation_mae(real_df[num_present], synthetic_df[num_present]),
    }
    report["mean_kl"] = float(np.nanmean(list(report["kl_divergences"].values())))

    logger.info(
        f"  MMD={report['mmd']:.6f} corr_MAE={report['correlation_mae']:.4f} "
        f"mean_KL={report['mean_kl']:.4f}"
    )
    return report


def fidelity_table(reports: dict[str, dict]) -> pd.DataFrame:
    """One row per method with the three headline fidelity numbers."""
    return pd.DataFrame(
        [
            {
                "method": r["method"],
                "n_synthetic": r["n_synthetic"],
                "mmd": r["mmd"],
                "correlation_mae": r["correlation_mae"],
                "mean_kl": r["mean_kl"],
            }
            for r in reports.values()
        ]
    )


def kl_table(reports: dict[str, dict], columns: list) -> pd.DataFrame:
    """Per column KL divergence with one column per method, for the columns given."""
    data = {
        method: {col: report["kl_divergences"].get(col, float("nan")) for col in columns}
        for method, report in reports.items()
    }
    return pd.DataFrame(data)


def _kl_divergences(real_df, synthetic_df, cat_cols: list, eps: float = 1e-8) -> dict[str, float]:
    """KL divergence of the real category frequencies against the synthetic ones, per column."""
    scores = {}
    for col in cat_cols:
        categories = sorted(
            set(real_df[col].dropna().unique()) | set(synthetic_df[col].dropna().unique())
        )
        real_counts = real_df[col].value_counts()
        synth_counts = synthetic_df[col].value_counts()

        p = np.array([real_counts.get(c, 0) for c in categories], dtype=float) + eps
        q = np.array([synth_counts.get(c, 0) for c in categories], dtype=float) + eps
        p /= p.sum()
        q /= q.sum()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores[col] = float(np.sum(kl_div(p, q)))

    return scores


def _numerical_statistics(real_df, synthetic_df, num_cols: list) -> pd.DataFrame:
    """Mean and standard deviation per numerical column, real against synthetic."""
    rows = []
    for col in num_cols:
        real = real_df[col].dropna()
        synth = synthetic_df[col].dropna()
        rows.append(
            {
                "column": col,
                "real_mean": float(real.mean()),
                "synthetic_mean": float(synth.mean()),
                "real_std": float(real.std()),
                "synthetic_std": float(synth.std()),
            }
        )
    return pd.DataFrame(rows)


def _correlation_mae(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    """Mean absolute difference between the real and synthetic Spearman correlation matrices."""
    real_corr = real_df.corr(method="spearman").to_numpy()
    synth_corr = synthetic_df.corr(method="spearman").to_numpy()
    upper = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    return float(np.nanmean(np.abs(real_corr[upper] - synth_corr[upper])))
