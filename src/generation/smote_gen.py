from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from src.utils.logging import get_logger

logger = get_logger(__name__)



# ---- Public functions --------------------------------------------------------
def generate_smote(
    train_df: pd.DataFrame,
    cfg: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Apply SMOTENC oversampling to ``train_df`` and return a balanced dataset.

    Note: unlike the SDV-based generators, SMOTE does not use metadata.
    It requires only the DataFrame and the list of categorical column indices.

    Parameters
    ----------
    train_df : pd.DataFrame
        Real training data (id columns dropped, target column included).
    cfg : dict
        Full project configuration.
    output_path : str or Path
        Where to write the synthetic CSV.

    Returns
    -------
    pd.DataFrame
        Resampled dataset. Contains all original rows plus the new
        synthetic minority-class rows. The class ratio will be 1:1.

    Raises
    ------
    ValueError
        If the target column is not binary.
    """
    target_col   = cfg["dataset"]["target_column"]
    cat_cols_cfg = set(cfg["dataset"].get("categorical_columns", []))
    k_neighbors  = cfg["generation"]["methods"]["smote"]["k_neighbors"]
    seed         = cfg["seed"]

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    if y.nunique() != 2:
        raise ValueError(
            f"SMOTE requires a binary target; '{target_col}' has {y.nunique()} unique values."
        )

    # Encode categoricals for SMOTENC
    # SMOTENC needs the positions (integer indices) of categorical columns.
    X_encoded, encoders = _encode_categoricals(X, cat_cols_cfg)
    cat_indices = _get_cat_indices(X_encoded, cat_cols_cfg)

    logger.info(
        f"SMOTE: fitting on {len(X_encoded)} rows "
        f"(k_neighbors={k_neighbors}, seed={seed})..."
    )
    
    logger.info(f"Categorical column indices: {cat_indices[:10]}")

    # NOTE: SMOTENC requires no NaN values.
    X_encoded_filled, fill_values = _fill_nans_for_smote(X_encoded)

    logger.info(f"SMOTE: filled NaNs in {len(fill_values)} columns for kNN computation.")
    for col, val in fill_values.items():
        logger.debug(f"  {col}: filled NaNs with {val}")

    smote = SMOTENC(
        categorical_features=cat_indices,
        k_neighbors=k_neighbors,
        random_state=seed,
    )

    X_resampled, y_resampled = smote.fit_resample(X_encoded_filled, y)

    # Decode categoricals back to original string labels
    X_decoded = _decode_categoricals(X_resampled, X.columns.tolist(), encoders)

    synthetic_df = X_decoded.copy()
    synthetic_df[target_col] = y_resampled.values

    # Preserve original column order
    synthetic_df = synthetic_df[train_df.columns]

    logger.info(
        f"SMOTE: resampled to {len(synthetic_df)} rows  (original: {len(train_df)})"
    )
    logger.info(
        f"Class distribution after resampling: {synthetic_df[target_col].value_counts().to_dict()}"
    )

    _save(synthetic_df, output_path)
    return synthetic_df



# ---- Internal helpers --------------------------------------------------------
def _encode_categoricals(
    X: pd.DataFrame,
    cat_cols: set[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Label-encode categorical columns to integers (required by SMOTENC).

    Returns the encoded DataFrame and a dict of per-column encoders
    (label -> int mappings) needed for decoding.
    """
    X_enc = X.copy()
    encoders: dict[str, dict] = {}

    for col in X.columns:
        if col in cat_cols or X[col].dtype == object:
            # Build a stable label -> int mapping (sorted for reproducibility)
            unique_labels = sorted(X[col].dropna().unique().tolist())
            label_to_int  = {lbl: i for i, lbl in enumerate(unique_labels)}
            int_to_label  = {i: lbl for lbl, i in label_to_int.items()}

            # Encode; use -1 for NaN (SMOTE will treat it as a valid code)
            X_enc[col] = X[col].map(label_to_int).fillna(-1).astype(int)
            encoders[col] = {"label_to_int": label_to_int, "int_to_label": int_to_label}

    return X_enc, encoders


def _get_cat_indices(X_encoded: pd.DataFrame, cat_cols: set[str]) -> list[int]:
    """Return the integer column positions of all categorical columns."""
    return [
        i
        for i, col in enumerate(X_encoded.columns)
        if col in cat_cols or X_encoded[col].dtype == object
    ]


def _decode_categoricals(
    X_resampled: np.ndarray | pd.DataFrame,
    columns: list[str],
    encoders: dict[str, dict],
) -> pd.DataFrame:
    """
    Reverse the label encoding applied in ``_encode_categoricals``.
    SMOTENC may interpolate integer codes; we round and clip before decoding.
    """
    if isinstance(X_resampled, np.ndarray):
        df = pd.DataFrame(X_resampled, columns=columns)
    else:
        df = X_resampled.copy()
        df.columns = columns

    for col in columns:
        if col in encoders:
            int_to_label = encoders[col]["int_to_label"]
            max_code = max(int_to_label.keys())
            # Round interpolated codes to nearest integer, clip to valid range
            codes = df[col].round().clip(lower=0, upper=max_code).astype(int)
            df[col] = codes.map(int_to_label)

    return df


def _save(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"SMOTE: synthetic dataset saved -> {path}  ({len(df)} rows)")


def _fill_nans_for_smote(
    X_encoded: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill NaN values so SMOTENC can compute kNN distances.

    Strategy:
    - Numerical columns with NaN: fill with the column median.
    - Integer-encoded categorical columns (encoded with -1 for NaN by
      _encode_categoricals): replace -1 with 0 (the first valid label code).

    This is a local, SMOTE-internal operation. It does NOT modify the
    original training DataFrame and does not constitute general dataset
    imputation. We only fill NaNs temporarily for the kNN distance computation.
    The resulting synthetic rows are brand-new interpolated points, not imputed
    real rows.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (filled DataFrame, {column_name: fill_value_used})
    """
    X_filled = X_encoded.copy()
    fill_values: dict = {}

    for col in X_filled.columns:
        col_dtype = X_filled[col].dtype
        # Categorical columns are encoded to integers with -1 for NaN
        if col_dtype in ("int64", "int32", int) or str(col_dtype).startswith("int"):
            mask = X_filled[col] == -1
            if mask.any():
                X_filled.loc[mask, col] = 0
                fill_values[col] = 0
        else:
            if X_filled[col].isnull().any():
                median_val = float(X_filled[col].median())
                X_filled[col] = X_filled[col].fillna(median_val)
                fill_values[col] = median_val

    return X_filled, fill_values