from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from src.utils.logging import get_logger
logger = get_logger(__name__)



def get_feature_columns(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    exclude_missingness_flags: bool = False,
) -> list[str]:
    """Return feature columns, excluding both target columns, performance scores and id columns.
 
    exclude_missingness_flags: also drop the `{attr}_was_nan` columns G1 adds
    for protected attributes it never imputes (intentionally). Those flags are proxies for the corresponding
    protected attribute's missingness, which can itself correlate with the
    attribute's value 
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
    features = [
        c for c in df.columns
        if c not in exclude
        and not (exclude_missingness_flags and c.endswith("_was_nan"))
    ]
    logger.debug("Feature columns (%d): %s", len(features), features)
    return features
