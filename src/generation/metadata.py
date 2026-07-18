from __future__ import annotations

from typing import Any

import pandas as pd
from sdv.metadata import SingleTableMetadata

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_metadata(df: pd.DataFrame, cfg: dict[str, Any]) -> SingleTableMetadata:
    """Build SDV metadata, marking a column categorical if it is listed as such, is object dtype, or has few unique values."""
    explicit_categoricals = set(cfg["dataset"].get("categorical_columns", []))
    # Few-valued integer columns (e.g. level_MAT: 1-4) are categorical too.
    MAX_UNIQUE_FOR_CATEGORICAL = 20

    metadata = SingleTableMetadata()

    column_type_counts = {"categorical": 0, "numerical": 0}

    for col in df.columns:
        if col in explicit_categoricals or df[col].dtype == object:
            sdtype = "categorical"
        elif df[col].nunique(dropna=True) <= MAX_UNIQUE_FOR_CATEGORICAL:
            sdtype = "categorical"
        else:
            sdtype = "numerical"

        metadata.add_column(column_name=col, sdtype=sdtype)
        column_type_counts[sdtype] += 1

    logger.info(
        f"Metadata built: {column_type_counts['categorical']} categorical, "
        f"{column_type_counts['numerical']} numerical columns"
    )

    metadata.validate()
    return metadata