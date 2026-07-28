from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from src.generation.io import save_synthetic
from src.generation.metadata import categorical_columns
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_smote(
    train_df: pd.DataFrame,
    cfg: dict,
    output_path
) -> pd.DataFrame:
    """Oversample the minority class with SMOTENC to a 1:1 balance.

    SMOTENC cannot operate on missing values, so they are filled for the nearest neighbour
    search and then restored on the original rows.
    """
    target_col = cfg["dataset"]["target_column"]
    k_neighbors = cfg["generation"]["methods"]["smote"]["k_neighbors"]
    seed = cfg["seed"]

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Get categorical columns using the same rule the SDV generators use
    cat_cols = set(categorical_columns(X, cfg))

    X_encoded, encoders = _encode_categoricals(X, cat_cols)
    cat_indices = [i for i, col in enumerate(X_encoded.columns) if col in encoders]

    logger.info(f"SMOTE: fitting on {len(X_encoded)} rows (k_neighbors={k_neighbors}, seed={seed})")
    X_filled = _fill_for_neighbour_search(X_encoded, encoders)

    smote = SMOTENC(categorical_features=cat_indices, k_neighbors=k_neighbors, random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X_filled, y)

    X_decoded = _decode_categoricals(X_resampled, X.columns.tolist(), encoders)
    synthetic_df = X_decoded.copy()
    synthetic_df[target_col] = np.asarray(y_resampled)
    
    n_original = len(X)
    if not np.array_equal(synthetic_df[target_col].to_numpy()[:n_original], y.to_numpy()):
        raise RuntimeError("SMOTENC did not return the original rows first, cannot restore missingness.")

    # Preserve original column order and restore missing values on the original rows
    synthetic_df = _restore_original_missing(synthetic_df, X)
    synthetic_df = synthetic_df[train_df.columns]

    n_new = len(synthetic_df) - n_original
    logger.info(f"SMOTE: {n_original} original rows plus {n_new} interpolated rows")
    logger.info(f"Class balance: {synthetic_df[target_col].value_counts().to_dict()}")

    save_synthetic(synthetic_df, output_path, "SMOTE")
    return synthetic_df


def missingness_comparison(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Per column missingness rate before and after oversampling."""
    real_rate = real_df.isna().mean()
    synth_rate = synthetic_df.isna().mean()
    comparison = pd.DataFrame({"real": real_rate, "synthetic": synth_rate})
    comparison["difference"] = comparison["synthetic"] - comparison["real"]
    return comparison[comparison["real"] > 0].sort_values("real", ascending=False)


def _encode_categoricals(X: pd.DataFrame, cat_cols: set) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns to integers, returning the frame and the mappings."""
    X_encoded = X.copy()
    encoders: dict[str, dict] = {}

    for col in X.columns:
        if col in cat_cols or X[col].dtype == object:
            labels = sorted(X[col].dropna().unique().tolist())
            label_to_int = {label: i for i, label in enumerate(labels)}
            X_encoded[col] = X[col].map(label_to_int).fillna(-1).astype(int)
            encoders[col] = {i: label for label, i in label_to_int.items()}

    return X_encoded, encoders


def _fill_for_neighbour_search(X_encoded: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Fill missing values so SMOTENC can compute distances."""
    X_filled = X_encoded.copy()

    for col in X_filled.columns:
        if col in encoders:
            known = X_filled.loc[X_filled[col] != -1, col]
            X_filled.loc[X_filled[col] == -1, col] = known.mode().iloc[0]
        elif X_filled[col].isna().any():
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())

    return X_filled


def _decode_categoricals(X_resampled, columns: list, encoders: dict) -> pd.DataFrame:
    """Reverse the label encoding, rounding interpolated codes back to valid labels."""
    df = pd.DataFrame(np.asarray(X_resampled), columns=columns)

    for col, mapping in encoders.items():
        codes = df[col].round().clip(lower=0, upper=max(mapping)).astype(int)
        df[col] = codes.map(mapping)

    return df


def _restore_original_missing(synthetic_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Put the original missing values back on the rows that were copied rather than interpolated."""
    restored = synthetic_df.copy()
    n_total = len(restored)

    for col in X.columns:
        missing = X[col].isna().to_numpy()
        if not missing.any():
            continue
        full_mask = np.zeros(n_total, dtype=bool)
        full_mask[: len(missing)] = missing
        restored.loc[full_mask, col] = np.nan

    return restored
