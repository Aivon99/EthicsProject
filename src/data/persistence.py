from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_dir(path) -> Path:
    """Create a directory and its parents if they do not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_csv(df: pd.DataFrame, path):
    """Save a DataFrame to CSV, creating parent directories."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")
