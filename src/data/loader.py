from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_config

logger = get_logger(__name__)

def download_dataset(dest = None, *, force = False):
    cfg = load_config()
    dest = Path(dest) if dest else Path(cfg["paths"]["raw_data"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info(f"Raw data already present at {dest} – skipping download.")
        return dest

    zenodo_url = cfg["dataset"]["zenodo_url"]
    
    logger.info(f"Downloading from {zenodo_url} …")
    file_url = f"{zenodo_url}/files/aequitas_education.csv?download=1"
    urllib.request.urlretrieve(file_url, dest)
    logger.info(f"Dataset saved to {dest}.")
    
    return dest 
 
def load_raw(path = None, confirmDownload = False):
    cfg = load_config()
    path = Path(path) if path else Path(cfg["paths"]["raw_data"])

    if not path.exists():
        if confirmDownload:
            logger.warning("Raw data not found locally – attempting download.")
            download_dataset(path)
        else: 
            download = input(f"Raw data not found at {path}. Do you want to download it now? [y/N] ").strip().lower()
            if download == "y":
                logger.warning("Raw data not found locally – attempting download.")    
                download_dataset(path)
            else:
                raise FileNotFoundError(f"Raw data not found at {path} and download declined by user.")

    logger.info(f"Loading raw data from {path}.")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df

