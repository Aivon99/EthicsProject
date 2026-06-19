from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)



def get_feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Return the feature column names to use for training and prediction.

    Excluded columns:
    - target column
    - performance score / level columns (they leak the target)
    - administrative ID columns

    Parameters
    ----------
    df:
        The DataFrame whose columns are inspected.
    cfg:
        Parsed configuration dict.

    Returns
    -------
    list of column names ordered as they appear in *df*.
    """
    exclude = set(
        [cfg["dataset"]["target_column"]]
        + cfg["dataset"]["performance_columns"]
        + cfg["dataset"]["id_columns"]
    )
    features = [c for c in df.columns if c not in exclude]
    logger.debug("Feature columns (%d): %s", len(features), features)
    return features
