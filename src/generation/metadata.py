from __future__ import annotations

from typing import Any

import pandas as pd
from sdv.metadata import SingleTableMetadata

from src.utils.logging import get_logger

logger = get_logger(__name__)


MAX_UNIQUE_FOR_CATEGORICAL = 20


def categorical_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Columns every generator must treat as categorical rather than continuous.

    Both the SDV generators and SMOTENC read this list, so the four techniques are compared
    on the same description of the table.
    """
    declared = set(cfg["dataset"].get("categorical_columns", []))
    return [
        col for col in df.columns
        if col in declared
        or df[col].dtype == object
        or df[col].nunique(dropna=True) <= MAX_UNIQUE_FOR_CATEGORICAL
    ]


def build_metadata(df: pd.DataFrame, cfg: dict[str, Any]) -> SingleTableMetadata:
    """Describe the table for SDV, using the shared categorical rule."""
    categoricals = set(categorical_columns(df, cfg))

    metadata = SingleTableMetadata()

    column_type_counts = {"categorical": 0, "numerical": 0}

    for col in df.columns:
        sdtype = "categorical" if col in categoricals else "numerical"
        metadata.add_column(column_name=col, sdtype=sdtype)
        column_type_counts[sdtype] += 1

    logger.info(
        f"Metadata built: {column_type_counts['categorical']} categorical, "
        f"{column_type_counts['numerical']} numerical columns"
    )

    metadata.validate()
    return metadata