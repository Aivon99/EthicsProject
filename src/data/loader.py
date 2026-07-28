from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_data(file_path, download_url: str | None = None) -> pd.DataFrame:
    """Load the raw dataset, downloading it first if it is not present locally."""
    path = Path(file_path)
    if not path.exists():
        if not download_url:
            raise FileNotFoundError(f"{path} not found and no download url was given.")
        _download_from_gdrive(download_url, path)
    return pd.read_csv(path, low_memory=False)


def load_dataset(path, cfg: dict[str, Any], drop_id_columns: bool = True) -> pd.DataFrame:
    """Load a processed CSV, check the target is present and optionally drop the id columns."""
    path = Path(path)
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows and {df.shape[1]} columns")

    target = cfg["dataset"]["target_column"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}.")

    if drop_id_columns:
        id_cols = [c for c in cfg["dataset"]["id_columns"] if c in df.columns]
        if id_cols:
            df = df.drop(columns=id_cols)
            logger.info(f"Dropped {len(id_cols)} id columns")

    return df


def load_real_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the real train and test splits produced by G1."""
    train_path = cfg["paths"]["train_data"]
    test_path = cfg["paths"]["test_data"]
    logger.info(f"Loading real splits from {train_path} and {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_synthetic_dataset(cfg: dict[str, Any], method: str) -> pd.DataFrame:
    """Load one synthetic dataset produced by G2."""
    subdir = cfg["generation"]["output_subdirs"][method]
    filename = cfg["generation"]["output_filename_template"].format(method=subdir)
    path = cfg["paths"]["synthetic_dir"] / subdir / filename

    logger.info(f"Loading synthetic data [{method}] from {path}")
    return pd.read_csv(path)


def _download_from_gdrive(url: str, dest: Path) -> None:
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"{dest} not found locally, downloading from {url}")
    gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
    if not dest.exists():
        raise RuntimeError(f"Download from {url} did not produce {dest}.")
