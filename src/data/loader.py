from __future__ import annotations

from pathlib import Path
from typing import Any
import sys
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---- Public functions 

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)



def load_dataset(
    path: str | Path,
    cfg: dict[str, Any],
    *,
    drop_id_columns: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV dataset and perform basic schema validation.

    Parameters
    ----------
    path : str or Path
        Location of the CSV file.
    cfg : dict
        Configuration dict from ``load_config()``.
    drop_id_columns : bool, default True
        If True, columns listed under ``cfg["dataset"]["id_columns"]``
        are dropped before returning. Identifier columns carry no
        distributional signal and should not be passed to synthesizers.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If the target column or protected attributes are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded  %d rows x %d columns", *df.shape)

    _validate_schema(df, cfg)

    if drop_id_columns:
        id_cols = [c for c in cfg["dataset"]["id_columns"] if c in df.columns]
        if id_cols:
            df = df.drop(columns=id_cols)
            logger.info("Dropped %d id columns: %s", len(id_cols), id_cols)

    return df


def describe_dataset(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """
    Print a concise human-readable summary of the dataset.

    Covers: shape, dtypes, missing-value rates, and value-count summaries
    for each protected attribute. Useful as the first cell of any notebook
    that loads data.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : dict
    """
    target = cfg["dataset"]["target_column"]
    protected = cfg["dataset"]["protected_attributes"]

    print(f"{'=' * 60}")
    print(f"  Dataset summary")
    print(f"{'=' * 60}")
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Target: '{target}'  ->  {df[target].value_counts().to_dict()}")
    print()

    # Missing value overview
    missing = df.isnull().mean().sort_values(ascending=False)
    missing_nonzero = missing[missing > 0]
    if missing_nonzero.empty:
        print("  Missing values: none")
    else:
        print(f"  Missing values ({len(missing_nonzero)} columns with > 0% missing):")
        for col, rate in missing_nonzero.items():
            print(f"    {col:<55} {rate:.1%}")
    print()

    # Protected attributes
    present_protected = [c for c in protected if c in df.columns]
    absent_protected  = [c for c in protected if c not in df.columns]

    print(f"  Protected attributes ({len(present_protected)} present"
          + (f", {len(absent_protected)} absent" if absent_protected else "") + "):")
    for col in present_protected:
        n_missing = df[col].isnull().sum()
        if df[col].dtype == "object" or df[col].nunique() <= 10:
            vc = df[col].value_counts(dropna=False).head(5).to_dict()
            print(f"    {col:<55} {vc}  [missing={n_missing}]")
        else:
            print(f"    {col:<55} mean={df[col].mean():.3f}  std={df[col].std():.3f}"
                  f"  [missing={n_missing}]")
    if absent_protected:
        logger.warning(f"    Absent protected attributes: {absent_protected}")
    print(f"{'=' * 60}")



# ---- Internal helpers --------------------------------------------------------
def _validate_schema(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Raise ValueError if required columns are missing from df."""
    target = cfg["dataset"]["target_column"]
    protected = cfg["dataset"]["protected_attributes"]

    errors: list[str] = []

    if target not in df.columns:
        errors.append(f"Target column '{target}' not found in dataset.")

    missing_protected = [c for c in protected if c not in df.columns]
    if missing_protected:
        logger.warning(
            f"{len(missing_protected)} protected attributes not found in dataset: {missing_protected}"
        )

    if errors:
        raise ValueError("\n".join(errors))
        