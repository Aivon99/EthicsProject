from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import get_logger, load_config

logger = get_logger(__name__)


@dataclass #NOTE
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    protected_train: pd.DataFrame
    protected_test: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""
    protected_attrs: list[str] = field(default_factory=list)
    encoders: dict[str, Any] = field(default_factory=dict)
    scaler: Any = None

def save_split(split: DataSplit, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(split, f)
    logger.info(f"DataSplit saved to {path}.")


def load_split(path) -> DataSplit:
    with open(path, "rb") as f:
        split = pickle.load(f)
    logger.info(f"DataSplit loaded from {path}.")
    return split