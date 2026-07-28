from __future__ import annotations

import torch
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def seed_generator(cfg: dict) -> int:
    """Seed every source of randomness to ensure reproducibility."""

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def resolve_n_samples(cfg: dict, train_df: pd.DataFrame) -> int:
    """Number of rows to sample; null in the config means match the training set."""
    n = cfg["generation"]["n_synthetic_samples"]
    return len(train_df) if n is None else int(n)


def save_synthetic(df: pd.DataFrame, path, method: str) -> None:
    """Save generated synthetic dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"{method}: saved {len(df)} rows to {path}")
