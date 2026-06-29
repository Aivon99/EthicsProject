from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer

from src.evaluation.fairness import _binarise_attribute
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---- Public functions --------------------------------------------------------

def fit_predict_equalized_odds(
    clf: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    attr_train: pd.Series,
    X_test: pd.DataFrame,
    attr_test: pd.Series,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap ``clf`` with fairlearn's ThresholdOptimizer (post-processing,
    equalized-odds constraint), trained against a single protected attribute.

    Group membership uses the same median-split/2-value binarisation as
    ``fairness.py`` (:func:`_binarise_attribute`), so the privileged/
    unprivileged grouping mitigation optimises against matches the grouping
    later used to *measure* DPD/EOD/DI on the same test set.

    Returns
    -------
    (y_pred, y_prob):
        Hard predictions on ``X_test``. ThresholdOptimizer's group-aware
        decision rule has no calibrated probability output, so ``y_prob`` is
        the hard label itself -- AUC/Brier on mitigated rows are therefore a
        coarser (but still valid) score than on unmitigated rows.
    """
    eo_cfg = cfg.get("mitigation", {}).get("equalized_odds", {})
    seed = cfg.get("mitigation", {}).get("random_state") or cfg.get("seed")

    group_train = _binarise_attribute(attr_train).astype(int)
    group_test = _binarise_attribute(attr_test).astype(int)

    optimizer = ThresholdOptimizer(
        estimator=clf,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        grid_size=eo_cfg.get("grid_size", 1000),
        predict_method="predict_proba",
        prefit=False,
    )
    optimizer.fit(X_train, y_train, sensitive_features=group_train)
    y_pred = np.asarray(optimizer.predict(X_test, sensitive_features=group_test, random_state=seed))
    return y_pred, y_pred.astype(float)


def fit_predict_prejudice_remover(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    attr_train: pd.Series,
    X_test: pd.DataFrame,
    attr_test: pd.Series,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit aif360's PrejudiceRemover (in-processing) against a single
    protected attribute and return (y_pred, y_prob) on ``X_test``.

    PrejudiceRemover (Kamishima et al. 2012) is its own regularised
    logistic-regression-style model -- unlike the equalized-odds wrapper
    above, it is not parameterised by a base classifier; ``eta`` (config:
    ``mitigation.prejudice_remover.eta``) controls the fairness/accuracy
    trade-off.

    Uses the same :func:`_binarise_attribute` grouping as the rest of the
    pipeline for consistency with how fairness is measured afterwards.
    """
    from aif360.algorithms.inprocessing import PrejudiceRemover

    pr_cfg = cfg.get("mitigation", {}).get("prejudice_remover", {})
    eta = pr_cfg.get("eta", 1.0)

    group_train = _binarise_attribute(attr_train).astype(float)
    group_test = _binarise_attribute(attr_test).astype(float)

    attr_name = "sensitive"
    label_name = "label"

    train_bld = _make_binary_label_dataset(X_train, y_train, group_train, attr_name, label_name)
    test_placeholder_y = pd.Series(np.zeros(len(X_test)), index=X_test.index)
    test_bld = _make_binary_label_dataset(X_test, test_placeholder_y, group_test, attr_name, label_name)

    model = PrejudiceRemover(eta=eta, sensitive_attr=attr_name, class_attr=label_name)
    model.fit(train_bld)
    pred_bld = model.predict(test_bld)

    y_pred = pred_bld.labels.ravel().astype(int)
    y_prob = pred_bld.scores.ravel().astype(float)
    return y_pred, y_prob


# ---- Internal helpers --------------------------------------------------------

def _make_binary_label_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    group: pd.Series,
    attr_name: str,
    label_name: str,
):
    from aif360.datasets import BinaryLabelDataset

    df = X.copy()
    df[attr_name] = np.asarray(group)
    df[label_name] = np.asarray(y)
    return BinaryLabelDataset(
        df=df,
        label_names=[label_name],
        protected_attribute_names=[attr_name],
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[np.array([1.0])],
        unprivileged_protected_attributes=[np.array([0.0])],
    )
