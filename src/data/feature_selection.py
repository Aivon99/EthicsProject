from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from src.utils.logging import get_logger

logger = get_logger(__name__)

_METHODS = ("mutual_info", "f_classif", "variance")


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    protected_attrs: list[str],
    cfg: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    """Select the most informative non-sensitive features for downstream modelling.

    Why this exists
    ----------------
    ``preprocess()`` keeps a large number of raw, never-aggregated survey-item
    columns (mostly from the principal/teacher questionnaires) that pass
    through untouched because they aren't covered by any rename/aggregation
    rule. Feeding all of them to classifiers and synthetic-data generators
    (CTGAN/TVAE) adds noise and dimensionality without adding signal. This
    function scores every *non-sensitive* candidate column's relevance to the
    target and keeps only the top ``k``, while always keeping every
    configured protected attribute regardless of its score -- they're needed
    for fairness auditing whether or not they're individually predictive.

    Fit on the training split only (``X_train``/``y_train``) to avoid any
    test-set leakage into the selection decision.

    Controlled entirely via ``cfg["feature_selection"]``:

    .. code-block:: yaml

        feature_selection:
          enabled: true
          method: mutual_info   # mutual_info | f_classif | variance
          k: 60                 # number of non-sensitive columns to keep
          random_state: null    # falls back to cfg["seed"]

    Set ``enabled: false`` to disable entirely and keep every column
    (fully revertible -- no other code path changes).

    Parameters
    ----------
    X_train:
        Training feature matrix (already imputed; sensitive columns may
        still contain NaNs by design -- they are excluded from scoring).
    y_train:
        Training target (binary).
    protected_attrs:
        Columns to always keep, never score/drop.
    cfg:
        Parsed configuration dict.

    Returns
    -------
    selected_columns:
        ``protected_attrs`` + the top-``k`` scored candidates, in
        ``X_train``'s original column order.
    scores_df:
        One row per candidate column with its score and whether it was
        selected -- saved by the caller for transparency/manual review.
        Empty if feature selection is disabled.
    """
    fs_cfg = cfg.get("feature_selection", {})
    if not fs_cfg.get("enabled", False):
        logger.info("Feature selection disabled (feature_selection.enabled=False) -- keeping all columns.")
        return list(X_train.columns), pd.DataFrame()

    method = fs_cfg.get("method", "mutual_info")
    k = fs_cfg.get("k", 60)
    seed = fs_cfg.get("random_state") or cfg.get("seed")

    if method not in _METHODS:
        raise ValueError(f"Unknown feature_selection.method '{method}'. Valid options: {_METHODS}")

    candidates = [c for c in X_train.columns if c not in protected_attrs]
    if not candidates:
        logger.warning("No non-sensitive candidate columns to select from.")
        return list(X_train.columns), pd.DataFrame()

    X_cand = X_train[candidates].copy()

    # Encode categoricals for scoring purposes only -- this does not affect
    # the dtype/encoding of the columns actually returned to the caller.
    cat_cols = X_cand.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cand[cat_cols] = encoder.fit_transform(X_cand[cat_cols].astype(str))
    X_cand = X_cand.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if method == "variance":
        scores = X_cand.var().to_numpy()
    elif method == "f_classif":
        scores, _ = f_classif(X_cand.to_numpy(), y_train.to_numpy())
    else:  # mutual_info
        discrete_mask = [c in cat_cols for c in candidates]
        scores = mutual_info_classif(
            X_cand.to_numpy(), y_train.to_numpy(),
            discrete_features=discrete_mask, random_state=seed,
        )

    scores_df = (
        pd.DataFrame({"column": candidates, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    scores_df["selected"] = scores_df.index < k

    top_k = set(scores_df.loc[scores_df["selected"], "column"])
    selected = [c for c in X_train.columns if c in protected_attrs or c in top_k]

    logger.info(
        "Feature selection (%s): kept %d/%d non-sensitive candidates + %d protected attrs -> %d total columns.",
        method, len(top_k), len(candidates), len(protected_attrs), len(selected),
    )
    return selected, scores_df
