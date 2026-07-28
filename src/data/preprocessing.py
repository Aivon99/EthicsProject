from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Return feature columns, excluding both target columns, performance scores, id columns, and the missingness flags.

    The `{attribute}_was_nan` flags are always excluded.
    """
    excluded = (
        {task["column"] for task in cfg["tasks"].values()}
        | set(cfg["dataset"]["performance_columns"])
        | set(cfg["dataset"]["id_columns"])
    )
    features = [
        c for c in df.columns
        if c not in excluded and not c.endswith("_was_nan")
    ]
    logger.info(f"Feature set: {len(features)} columns")
    return features
