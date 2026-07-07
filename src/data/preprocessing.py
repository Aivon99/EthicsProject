from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)



def get_feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Return the feature column names to use for training and prediction.

    Excluded columns:
    - target column (both the active one and its sibling task's target,
      e.g. target_low_perf when target_high_perf is active, so one binary
      task never leaks into the other via a shared saved CSV)
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
    target_names = {
        cfg["dataset"]["target_column"],
        cfg["target"]["column_name"],
        cfg["target"].get("low_column_name"),
    }
    target_names.discard(None)
    exclude = (
        target_names
        | set(cfg["dataset"]["performance_columns"])
        | set(cfg["dataset"]["id_columns"])
    )
    features = [c for c in df.columns if c not in exclude]
    logger.debug("Feature columns (%d): %s", len(features), features)
    return features
