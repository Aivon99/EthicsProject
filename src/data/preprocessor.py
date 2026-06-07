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




class Preprocessor:
    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self._ds_cfg = self.cfg["dataset"]
        self.target_col = self._ds_cfg.get("target_column")
        self.protected_attrs = self._ds_cfg.get("protected_attrs", [])
        self.test_size = self._ds_cfg.get("test_size", 0.20)
        self.random_seed = self._ds_cfg.get("random_seed", 42)
        self._encoders: dict[str, LabelEncoder] = {}
        self._scaler: StandardScaler = None
        self._feature_names = []

    def fit_transform(self, df) -> DataSplit:
        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._infer_and_validate_columns(df)
        df = self._encode_categoricals(df, fit=True)

        X, y, protected = self._split_xy(df)

        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(X, y, protected, test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y,
        )

        X_train, X_test = self._scale(X_train, X_test, fit=True)
        split = DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            protected_train=prot_train.reset_index(drop=True),
            protected_test=prot_test.reset_index(drop=True),
            feature_names=self._feature_names,
            target_name=self.target_col,
            protected_attrs=self.protected_attrs,
            encoders=self._encoders,
            scaler=self._scaler,
        )


        logger.info( # WE gotta log it up (in we gotta pump it up rhythm)
            f"Split: {len(X_train):,} train / {len(X_test):,} test rows "
            f"| {len(self._feature_names)} features."
        )
        
        return split
    

    def transform(self, df):
        if not self._encoders and self._scaler is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        df = df.copy()
        df = self._encode_categoricals(df, fit=False)
        X, _, _ = self._split_xy(df)
        X_scaled, _ = self._scale(X, X, fit=False)
        return X_scaled
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        
        logger.info(f"Preprocessor saved to {path}.")


    @classmethod
    def load(cls, path) -> "Preprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        
        logger.info(f"Preprocessor loaded from {path}.")
        
        return obj

    
    def _drop_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} duplicate rows.")
        return df

    def _encode_categoricals(self, df, *, fit):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le
            else:
                le = self._encoders.get(col)
                if le is None:
                    raise KeyError(f"no fitted encoder for column '{col}'.")
                df[col] = le.transform(df[col].astype(str))
        return df


    def _split_xy(self, df) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        protected = df[self.protected_attrs] if self.protected_attrs else pd.DataFrame()
        drop_cols = [self.target_col] + self.protected_attrs
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[self.target_col]
        self._feature_names = X.columns.tolist()
        return X, y, protected

    def _scale(self, X_train, X_test, *, fit: bool):
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        if fit:
            self._scaler = StandardScaler()
            X_train[num_cols] = self._scaler.fit_transform(X_train[num_cols])
        else:
            X_train[num_cols] = self._scaler.transform(X_train[num_cols])
        X_test[num_cols] = self._scaler.transform(X_test[num_cols])
        return X_train.reset_index(drop=True), X_test.reset_index(drop=True)

