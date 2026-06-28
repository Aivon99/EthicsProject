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
    Generate a synthetic dataset via density-based SMOTE interpolation that
    PRESERVES the natural class ratio (does NOT balance classes).
 
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
        Synthetic dataset of size ``n_synthetic_samples`` (default: same size
        as the training set), with the same columns and approximately the same
        class ratio as ``train_df``.
 
    Raises
    ------
    ValueError
        If the target column is missing or not binary.
    """
    target_col   = cfg["dataset"]["target_column"]
    cat_cols_cfg = set(cfg["dataset"].get("categorical_columns", []))
    seed         = cfg["seed"]
    k_neighbors  = cfg["generation"]["methods"]["smote"]["k_neighbors"]
    n_synthetic  = _resolve_n_samples(cfg, train_df)
 
    # --- Validate target ---
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if train_df[target_col].nunique() != 2:
        raise ValueError(
            f"This SMOTE variant expects a binary target; "
            f"'{target_col}' has {train_df[target_col].nunique()} unique values."
        )
 
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
 
    # --- Determine per-class synthetic budget (preserve natural ratio) ---
    class_props = y.value_counts(normalize=True).to_dict()    # {class: fraction}
    logger.info(
        "SMOTE (density mode): real class ratio = %s",
        {k: f"{v:.1%}" for k, v in class_props.items()},
    )
 
    per_class_target = _allocate_counts(class_props, n_synthetic)
    logger.info("SMOTE: synthetic budget per class = %s", per_class_target)
 
    # --- Encode categoricals once for the whole frame ---
    X_encoded, encoders = _encode_categoricals(X, cat_cols_cfg)
    cat_indices = _get_cat_indices(X_encoded, cat_cols_cfg)
    X_filled, _ = _fill_nans_for_smote(X_encoded)
 
    # --- Interpolate within each class separately ---
    synthetic_parts: list[pd.DataFrame] = []
    for cls, target_n in per_class_target.items():
        if target_n <= 0:
            continue
        syn_X_cls = _densify_one_class(
            X_filled=X_filled,
            y=y,
            cls=cls,
            target_n=target_n,
            cat_indices=cat_indices,
            k_neighbors=k_neighbors,
            seed=seed,
        )
        syn_X_cls_decoded = _decode_categoricals(
            syn_X_cls, X.columns.tolist(), cat_cols_cfg, encoders
        )
        syn_X_cls_decoded[target_col] = cls
        synthetic_parts.append(syn_X_cls_decoded)
 
    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
 
    # Shuffle so classes are not block-ordered, then restore column order
    synthetic_df = (
        synthetic_df.sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)[train_df.columns]
    )
 
    logger.info(
        "SMOTE: generated %d synthetic rows  (real: %d)  synthetic class ratio = %s",
        len(synthetic_df),
        len(train_df),
        {k: f"{v:.1%}" for k, v in
         synthetic_df[target_col].value_counts(normalize=True).to_dict().items()},
    )
 
    _save(synthetic_df, output_path)
    return synthetic_df



# ---- Internal helpers --------------------------------------------------------
ef _densify_one_class(
    X_filled: pd.DataFrame,
    y: pd.Series,
    cls: Any,
    target_n: int,
    cat_indices: list[int],
    k_neighbors: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate ``target_n`` synthetic feature rows by SMOTE-interpolating *within*
    the samples belonging to class ``cls``.
 
    Mechanism
    ---------
    SMOTENC always interpolates within the minority class to reach a target
    count. We exploit this by constructing a 2-class problem in which:
      - the real class-``cls`` samples are the MINORITY (label 1), and
      - a set of ``n_cls + target_n`` dummy 'anchor' rows is the MAJORITY (label 0).
    We then ask SMOTENC to lift label 1 up to the majority size. SMOTENC
    interpolates ONLY among the label-1 (real class-``cls``) rows, so the
    synthesised tail is exactly ``target_n`` new interpolations of real
    class-``cls`` neighbours. The anchors are discarded.
 
    This gives density augmentation *of that class*, with no balancing across the
    real target.
 
    Returns
    -------
    pd.DataFrame
        ``target_n`` synthetic feature rows (still integer-encoded).
    """
    rng = np.random.default_rng(seed + hash(str(cls)) % 10_000)
 
    in_class = (y == cls).to_numpy()
    X_cls = X_filled.loc[in_class].reset_index(drop=True)
    n_cls = len(X_cls)
 
    if n_cls <= k_neighbors:
        # Not enough samples for kNN interpolation: fall back to jittered
        # resampling. Rare; logged for transparency.
        logger.warning(
            "SMOTE: class %s has only %d samples (<= k_neighbors=%d); "
            "falling back to jittered resampling for %d synthetic rows.",
            cls, n_cls, k_neighbors, target_n,
        )
        return _jittered_resample(X_cls, target_n, cat_indices, rng)
 
    dummy_majority_n = n_cls + target_n
    anchors = _make_anchor_rows(X_cls, dummy_majority_n, cat_indices, rng)
 
    combined_X = pd.concat([X_cls, anchors], ignore_index=True)
    combined_y = np.concatenate([
        np.ones(n_cls, dtype=int),               # real class-cls → oversampled
        np.zeros(dummy_majority_n, dtype=int),   # anchors → fixed majority
    ])
 
    smote = SMOTENC(
        categorical_features=cat_indices,
        k_neighbors=k_neighbors,
        sampling_strategy={1: dummy_majority_n},  # lift class 1 to match anchors
        random_state=seed,
    )
    res_X, res_y = smote.fit_resample(combined_X, combined_y)
 
    res_X = pd.DataFrame(res_X, columns=combined_X.columns)
    label1_rows = res_X.loc[res_y == 1].reset_index(drop=True)
    synthetic_rows = label1_rows.iloc[n_cls:].reset_index(drop=True)
 
    # Guard: ensure exactly target_n rows
    if len(synthetic_rows) > target_n:
        synthetic_rows = synthetic_rows.iloc[:target_n].reset_index(drop=True)
    elif len(synthetic_rows) < target_n:
        extra = _jittered_resample(
            X_cls, target_n - len(synthetic_rows), cat_indices, rng
        )
        synthetic_rows = pd.concat([synthetic_rows, extra], ignore_index=True)
 
    return synthetic_rows
 
 
def _make_anchor_rows(
    X_cls: pd.DataFrame,
    n: int,
    cat_indices: list[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create ``n`` dummy 'anchor' rows used only to set SMOTENC's oversampling
    target. They are never returned in the synthetic output. We build them as
    jittered copies of real class samples so they sit in a plausible region and
    do not distort kNN within the real class (they carry the opposite label).
    """
    return _jittered_resample(X_cls, n, cat_indices, rng)
 
 
def _jittered_resample(
    X_cls: pd.DataFrame,
    n: int,
    cat_indices: list[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample ``n`` rows from ``X_cls`` with replacement and add small Gaussian
    jitter to numerical columns. Categorical columns are copied unchanged.
    Used (a) to build anchors and (b) as a fallback when a class is too small
    for kNN interpolation.
    """
    if len(X_cls) == 0:
        return pd.DataFrame(np.zeros((n, X_cls.shape[1])), columns=X_cls.columns)
 
    idx = rng.integers(0, len(X_cls), size=n)
    sampled = X_cls.iloc[idx].reset_index(drop=True).copy()
 
    num_cols = [c for i, c in enumerate(X_cls.columns) if i not in cat_indices]
    for c in num_cols:
        std = X_cls[c].std()
        if std > 0:
            sampled[c] = sampled[c] + rng.normal(0, std * 0.01, size=n)
 
    return sampled

def _allocate_counts(class_props: dict, n_total: int) -> dict:
    """
    Allocate ``n_total`` synthetic rows across classes in proportion to their
    real frequencies, using the largest-remainder method so the parts sum
    exactly to n_total.
    """
    raw = {cls: prop * n_total for cls, prop in class_props.items()}
    floored = {cls: int(np.floor(v)) for cls, v in raw.items()}
    remainder = n_total - sum(floored.values())
 
    frac_sorted = sorted(raw.keys(), key=lambda c: raw[c] - floored[c], reverse=True)
    for i in range(remainder):
        floored[frac_sorted[i % len(frac_sorted)]] += 1
 
    return floored
 
 
def _resolve_n_samples(cfg: dict, train_df: pd.DataFrame) -> int:
    """Number of synthetic rows to generate (default: match training set size)."""
    n = cfg["generation"].get("n_synthetic_samples")
    if n is None:
        return len(train_df)
    return int(n)
 
 
def _encode_categoricals(
    X: pd.DataFrame,
    cat_cols: set[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Label-encode categorical columns to integers (required by SMOTENC).
    NaN is encoded as -1 (filled later by ``_fill_nans_for_smote``).
 
    Returns the encoded DataFrame and per-column encoders for decoding.
    """
    X_enc = X.copy()
    encoders: dict[str, dict] = {}
 
    for col in X.columns:
        if col in cat_cols or X[col].dtype == object:
            unique_labels = sorted(X[col].dropna().unique().tolist())
            label_to_int  = {lbl: i for i, lbl in enumerate(unique_labels)}
            int_to_label  = {i: lbl for lbl, i in label_to_int.items()}
 
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
    cat_cols: set[str],
    encoders: dict[str, dict],
) -> pd.DataFrame:
    """
    Reverse the label encoding applied in ``_encode_categoricals``.
    SMOTENC keeps categorical codes integral, but we round / clip defensively.
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
            codes = df[col].round().clip(lower=0, upper=max_code).astype(int)
            df[col] = codes.map(int_to_label)
 
    return df
 
 
def _fill_nans_for_smote(
    X_encoded: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill NaN values so SMOTENC can compute kNN distances.
 
    - Numerical columns with NaN → column median.
    - Integer-encoded categorical columns (with -1 for NaN) → 0 (first label).
 
    This is a *local, SMOTE-internal* operation. It does NOT impute the original
    dataset and does not violate the project's decision to preserve MNAR
    missingness (notebook 02, Section 2.3): NaNs are only filled temporarily for
    the kNN distance computation, and synthetic rows are new interpolated points.
 
    Returns (filled DataFrame, {column: fill_value}).
    """
    X_filled = X_encoded.copy()
    fill_values: dict = {}
 
    for col in X_filled.columns:
        col_dtype = str(X_filled[col].dtype)
        if col_dtype.startswith("int"):
            mask = X_filled[col] == -1
            if mask.any():
                X_filled.loc[mask, col] = 0
                fill_values[col] = 0
        else:
            if X_filled[col].isnull().any():
                median_val = float(X_filled[col].median())
                X_filled[col] = X_filled[col].fillna(median_val)
                fill_values[col] = median_val
 
    if fill_values:
        logger.info(
            "SMOTE: filled NaNs/-1s in %d columns for kNN computation "
            "(internal to SMOTE step only).",
            len(fill_values),
        )
    return X_filled, fill_values
 
 
def _save(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("SMOTE: synthetic dataset saved → %s  (%d rows)", path, len(df))