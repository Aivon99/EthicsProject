from __future__ import annotations
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils import get_logger

logger = get_logger(__name__)


class LogisticRegressionClassifier:
    """simple log. regression classifier"""

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



class XGBoostClassifier:
    """ XGBoost classifier, some def values for safety, basically wrapper fo xgboost API """
    
    name = "xgboost"

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42):
        self._model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
            eval_metric="logloss", verbosity=0,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

class ShallowNNClassifier:
    """Shallow NNclass implemtation in pytorch, with some regularizationand def. values"""
    name = "shallow_nn"

    def __init__(self, hidden_dims=None, dropout=0.2, epochs=50, batch_size=64, lr=1e-3, random_state=42):
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state #To be set from config.  
        self._net = None

    def _build_net(self, n_features):
        dims = [n_features] + self.hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(self.dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        self._net = self._build_net(X_t.shape[1])
        
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr) #NOTE ADAM OPTIMIZER
        
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        self._net.train()

        for i in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        self._net.eval()

        with torch.no_grad():
            X_t = torch.tensor(X.values, dtype=torch.float32)
            probs = torch.sigmoid(self._net(X_t).squeeze(1)).numpy()
        return np.stack([1 - probs, probs], axis=1)


    def predict(self, X):
        #Over threshold
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
