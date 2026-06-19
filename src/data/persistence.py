from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---- Public functions --------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Destination file path. Parent directories are created automatically.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    logger.info("Saved CSV to %s", p)
