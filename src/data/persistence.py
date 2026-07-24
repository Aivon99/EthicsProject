from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)
#Nothing particularly interesting just helpers mildly misplaced

def ensure_dir(path):
    """Create path (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df, path):
    """Save a DataFrame to CSV, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    logger.info("Saved CSV to %s", p)
