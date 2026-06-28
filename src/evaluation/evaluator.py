from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.utility import compute_clf_metrics, maximum_mean_discrepancy, column_correlation_delta, utility_delta
from src.evaluation.fairness import compute_fairness_metrics, fairness_delta
from src.models.classifiers import build_classifier
from src.data.preprocessor import DataSplit
from src.utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, split: DataSplit, cfg: dict, protected_attrs = None):
        self.split = split
        self.cfg = cfg
        self.protected_attrs = protected_attrs or split.protected_attrs
        self._real_metrics: dict | None = None
        self._real_fairness: dict | None = None

    def baseline(self, classifier_name) -> dict:
        logger.info(f"Computing real-data baseline for [{classifier_name}] …")
        real_result = self._run(self.split.X_train, self.split.y_train, classifier_name, "real")
        self._real_metrics = {k: real_result[k] for k in ("balanced_accuracy", "f1_macro", "roc_auc")}
        self._real_fairness = {k: v for k, v in real_result.items()
                               if any(a in k for a in self.protected_attrs)}
        return real_result

    def evaluate(
        self,
        X_synth: pd.DataFrame,
        y_synth: pd.Series,
        classifier_name,
        generator_name,
        repetition = 0,
        X_real_for_mmd = None,
    ) -> dict:
        if self._real_metrics is None:
            self.baseline(classifier_name)

        logger.info(f"Evaluating [{generator_name} | {classifier_name} | rep={repetition}] …")
        synth_result = self._run(X_synth, y_synth, classifier_name, "synthetic")

        synth_clf = {k: synth_result[k] for k in ("balanced_accuracy", "f1_macro", "roc_auc")}
        gap = utility_delta(self._real_metrics, synth_clf)

        X_real = X_real_for_mmd if X_real_for_mmd is not None else self.split.X_train
        mmd_val = maximum_mean_discrepancy(X_real, X_synth)
        corr = column_correlation_delta(X_real, X_synth)

        synth_fairness = {k: synth_result[k] for k in synth_result
                          if any(a in k for a in self.protected_attrs)}
        f_gap = fairness_delta(self._real_fairness, synth_fairness)

        return {
            "generator": generator_name,
            "classifier": classifier_name,
            "repetition": repetition,
            **synth_result,
            **gap,
            "mmd": mmd_val,
            "corr_mean_abs_delta": corr["mean_abs_delta"],
            "corr_max_abs_delta": corr["max_abs_delta"],
            **{f"fairness_gap_{k}": v for k, v in f_gap.items()},
        }

    def _run(self, X_train, y_train, classifier_name, label) -> dict:
        clf = build_classifier(classifier_name, self.cfg, y_train=y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(self.split.X_test)
        y_prob = clf.predict_proba(self.split.X_test)[:, 1]
        metrics = compute_clf_metrics(self.split.y_test.values, y_pred, y_prob)
        for attr in self.protected_attrs:
            if attr in self.split.protected_test.columns:
                metrics.update(compute_fairness_metrics(
                    self.split.y_test.values, y_pred,
                    self.split.protected_test[attr], attr_name=attr,
                ))
        logger.info(f"  [{label}] AUC={metrics['roc_auc']:.4f} | BA={metrics['balanced_accuracy']:.4f}")
        return metrics