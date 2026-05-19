from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils import get_logger

logger = get_logger(__name__)


class LogisticRegressionClassifier:
    name = "logistic_regression"

    def __init__(self, max_iter=1000, C=1.0, random_state=42):
        self._model = LogisticRegression(max_iter=max_iter, C=C, random_state=random_state)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

