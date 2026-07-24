from __future__ import annotations

from pathlib import Path
from typing import Any
import sys
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_data(file_path, download_url = None):
    """Load data from a CSV file, downloading it first if missing"""
    path = Path(file_path)
    if not path.exists():
        if not download_url:
            print(f"Error loading data: {path} not found and no download_url was provided.")
            sys.exit(1)
        try:
            _download_from_gdrive(download_url, path)
        except Exception as e:
            print(f"Error downloading data from {download_url}: {e}")
            sys.exit(1)

    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def _download_from_gdrive(url: str, dest: Path):
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("%s not found locally -- downloading from %s", dest, url)
    output = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
    if output is None or not dest.exists():
        raise RuntimeError(f"Download from {url} did not produce {dest}.")
    logger.info("Downloaded dataset to %s", dest)



def load_dataset(
    path,
    cfg,
    drop_id_columns = True,
) :
    """Load a CSV, validate the schema, and optionally drop id columns."""
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
    """Print shape, missing-value rates and protected-attribute value counts."""


    target = cfg["dataset"]["target_column"]
    protected = cfg["dataset"]["protected_attributes"]

    print(f"{'=' * 60}")
    print(f"  Dataset summary")
    print(f"{'=' * 60}")
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Target: '{target}'  ->  {df[target].value_counts().to_dict()}")
    print()

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


def load_real_data(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preprocessed real train and test splits."""
    train_path = Path(cfg["paths"]["train_data"])
    test_path  = Path(cfg["paths"]["test_data"])

    logger.info("Loading real train data from %s", train_path)
    logger.info("Loading real test data  from %s", test_path)

    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_synthetic_dataset(cfg, method):

    synthetic_dir     = Path(cfg["paths"]["synthetic_dir"])
    output_subdirs    = cfg["generation"]["output_subdirs"]
    filename_template = cfg["generation"]["output_filename_template"]

    subdir   = output_subdirs[method]
    filename = filename_template.format(method=subdir)
    path     = synthetic_dir / subdir / filename

    logger.info("Loading synthetic data [%s] from %s", method, path)
    return pd.read_csv(path)


def load_all_synthetic_datasets(cfg) -> dict[str, pd.DataFrame]:
    methods = list(cfg["generation"]["output_subdirs"].keys())
    return {m: load_synthetic_dataset(cfg, m) for m in methods}


def _validate_schema(df, cfg):
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
        