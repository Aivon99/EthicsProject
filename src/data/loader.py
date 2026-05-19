from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_config

logger = get_logger(__name__)

 
def load_raw(path = None) :
    cfg = load_config()
    path = Path(path) if path else Path(cfg["paths"]["raw_data"])

    if not path.exists():
        logger.warning("Raw data not found locally – attempting download.")
        download_dataset(path)

    logger.info(f"Loading raw data from {path}.")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df

