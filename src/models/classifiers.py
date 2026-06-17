from __future__ import annotations

from typing import Any

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_logistic_regression(cfg: dict[str, Any]) -> LogisticRegression:
    """Instantiate a Logistic Regression classifier from config."""
    lr_cfg = cfg["experiments"]["classifiers"]["logistic_regression"]
    clf = LogisticRegression(
        max_iter     = lr_cfg["max_iter"],
        C            = lr_cfg["C"],
        solver       = lr_cfg["solver"],
        class_weight = lr_cfg["class_weight"],
        random_state = cfg["seed"],
    )
    logger.debug("LogisticRegression params: %s", clf.get_params())
    return clf


def make_xgboost(
    cfg: dict[str, Any],
    y_train: pd.Series | None = None,
) -> XGBClassifier:
    """Instantiate an XGBoost classifier from config.

    Parameters
    ----------
    cfg:
        Parsed configuration dict.
    y_train:
        Optional training labels used to compute ``scale_pos_weight``
        automatically when the config value is ``"auto"``. If *None*, the
        weight defaults to 1 (no adjustment).
    """
    xgb_cfg = cfg["experiments"]["classifiers"]["xgboost"]

    # Compute class-imbalance weight if requested.
    scale_pos_weight = xgb_cfg["scale_pos_weight"]
    if scale_pos_weight == "auto":
        if y_train is not None:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
            logger.debug("XGBoost scale_pos_weight computed: %.3f", scale_pos_weight)
        else:
            scale_pos_weight = 1.0
            logger.warning(
                "scale_pos_weight='auto' but no y_train provided; defaulting to 1.0"
            )

    clf = XGBClassifier(
        n_estimators     = xgb_cfg["n_estimators"],
        max_depth        = xgb_cfg["max_depth"],
        learning_rate    = xgb_cfg["learning_rate"],
        subsample        = xgb_cfg["subsample"],
        colsample_bytree = xgb_cfg["colsample_bytree"],
        scale_pos_weight = scale_pos_weight,
        eval_metric      = xgb_cfg["eval_metric"],
        random_state     = cfg["seed"],
        verbosity        = 0,
    )
    logger.debug("XGBClassifier params: %s", clf.get_params())
    return clf


def make_mlp(cfg: dict[str, Any]) -> MLPClassifier:
    """Instantiate a shallow MLP classifier from config."""
    mlp_cfg = cfg["experiments"]["classifiers"]["mlp"]
    clf = MLPClassifier(
        hidden_layer_sizes   = tuple(mlp_cfg["hidden_layer_sizes"]),
        activation           = mlp_cfg["activation"],
        max_iter             = mlp_cfg["max_iter"],
        early_stopping       = mlp_cfg["early_stopping"],
        validation_fraction  = mlp_cfg["validation_fraction"],
        n_iter_no_change     = mlp_cfg["n_iter_no_change"],
        learning_rate_init   = mlp_cfg["learning_rate_init"],
        batch_size           = mlp_cfg["batch_size"],
        random_state         = cfg["seed"],
    )
    logger.debug("MLPClassifier params: %s", clf.get_params())
    return clf


# ---- Utilities ---------------------------------------------------------------

# Maps the config key to the factory function.
_CLASSIFIER_FACTORIES = {
    "logistic_regression": make_logistic_regression,
    "xgboost":             make_xgboost,
    "mlp":                 make_mlp,
}

# Display names for notebooks / plots.
CLASSIFIER_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost":             "XGBoost",
    "mlp":                 "MLP",
}

def build_classifier(
    name: str,
    cfg: dict[str, Any],
    y_train: pd.Series | None = None,
) -> Any:
    """Build and return a classifier by its config key.

    Parameters
    ----------
    name:
        One of ``"logistic_regression"``, ``"xgboost"``, ``"mlp"``.
    cfg:
        Parsed configuration dict.
    y_train:
        Training labels (only needed by XGBoost for ``scale_pos_weight``).

    Returns
    -------
    Unfitted sklearn-compatible estimator.
    """
    if name not in _CLASSIFIER_FACTORIES:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Valid options: {list(_CLASSIFIER_FACTORIES.keys())}"
        )
    factory = _CLASSIFIER_FACTORIES[name]
    if name == "xgboost":
        return factory(cfg, y_train=y_train)
    return factory(cfg)