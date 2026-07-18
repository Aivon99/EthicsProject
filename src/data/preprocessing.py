from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)



def get_feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Return feature columns, excluding both target columns, performance scores and id columns."""
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
