from __future__ import annotations

from typing import Any

import pandas as pd
from sdv.metadata import SingleTableMetadata

from src.utils.logging import get_logger

logger = get_logger(__name__)



# ---- Public functions --------------------------------------------------------
def build_metadata(df: pd.DataFrame, cfg: dict[str, Any]) -> SingleTableMetadata:
    """
    Construct an SDV ``SingleTableMetadata`` object from the DataFrame
    and the project configuration.

    The function auto-detects column types using the following rules,
    applied in priority order:

    1. If the column is in ``cfg["dataset"]["categorical_columns"]`` -> sdtype = "categorical"
    2. If dtype is ``object`` or has <= ``MAX_UNIQUE_FOR_CATEGORICAL`` unique
       values -> sdtype = "categorical"
    3. Otherwise -> sdtype = "numerical"

    SDV uses metadata to know:
        - which columns are categorical (requires one-hot / label encoding)
        - which are numerical (continuous or discrete)
        - which is the primary key (excluded from learning)
        - which are boolean

    Parameters
    ----------
    df : pd.DataFrame
        The training dataset (id columns already dropped).
    cfg : dict
        Configuration dict from ``load_config()``.

    Returns
    -------
    sdv.metadata.SingleTableMetadata
        Validated metadata object ready to pass to any SDV synthesizer.
    """
    explicit_categoricals = set(cfg["dataset"].get("categorical_columns", []))
    # Columns with few unique values are almost certainly categorical even if
    # stored as integers (e.g. level_MAT: 1, 2, 3, 4).
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