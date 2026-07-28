from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer

from src.evaluation.fairness import binarise_attribute
from src.utils.logging import get_logger

logger = get_logger(__name__)


def fit_predict_equalized_odds(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    attr_train: pd.Series,
    X_test: pd.DataFrame,
    attr_test: pd.Series,
    cfg: dict[str, Any],
):
    """Wrap a classifier in fairlearn's ThresholdOptimizer under an equalized odds constraint.

    ThresholdOptimizer has no calibrated probability output, so the returned probability is the
    hard label itself and AUC or Brier on these rows are coarser than on unmitigated rows.
    """
    seed = cfg["mitigation"]["random_state"] or cfg["seed"]
    group_train, known = _groups_for_mitigation(attr_train, cfg)
    group_test, _ = _groups_for_mitigation(attr_test, cfg)

    optimizer = ThresholdOptimizer(
        estimator=clf,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        grid_size=cfg["mitigation"]["equalized_odds"]["grid_size"],
        predict_method="predict_proba",
        prefit=False,
    )
    optimizer.fit(X_train[known], y_train[known], sensitive_features=group_train[known])
    y_pred = np.asarray(optimizer.predict(X_test, sensitive_features=group_test, random_state=seed))
    return y_pred, y_pred.astype(float)


def fit_predict_prejudice_remover(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    attr_train: pd.Series,
    X_test: pd.DataFrame,
    attr_test: pd.Series,
    cfg: dict[str, Any],
):
    """Fit aif360's PrejudiceRemover, an in-processing model whose loss penalises dependence on the attribute."""
    from aif360.algorithms.inprocessing import PrejudiceRemover

    group_train, known = _groups_for_mitigation(attr_train, cfg)
    group_test, _ = _groups_for_mitigation(attr_test, cfg)

    train_bld = _make_binary_label_dataset(
        X_train[known], y_train[known], group_train[known].astype(float),
    )
    placeholder_y = pd.Series(np.zeros(len(X_test)), index=X_test.index)
    test_bld = _make_binary_label_dataset(X_test, placeholder_y, group_test.astype(float))

    model = PrejudiceRemover(
        eta=cfg["mitigation"]["prejudice_remover"]["eta"],
        sensitive_attr="sensitive",
        class_attr="label",
    )
    model.fit(train_bld)
    predicted = model.predict(test_bld)

    return predicted.labels.ravel().astype(int), predicted.scores.ravel().astype(float)


def _groups_for_mitigation(attribute: pd.Series, cfg: dict[str, Any]):
    """Group labels for a mitigation library, plus a mask of the rows where the group is known."""
    binary = binarise_attribute(attribute, cfg)
    known = binary.notna().to_numpy()
    fallback = int(binary.mode(dropna=True).iloc[0])
    return binary.fillna(fallback).astype(int), known


def _make_binary_label_dataset(X: pd.DataFrame, y: pd.Series, group: pd.Series):
    from aif360.datasets import BinaryLabelDataset

    df = X.copy()
    df["sensitive"] = np.asarray(group)
    df["label"] = np.asarray(y)
    return BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=["sensitive"],
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[np.array([1.0])],
        unprivileged_protected_attributes=[np.array([0.0])],
    )
