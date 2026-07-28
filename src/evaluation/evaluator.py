from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessor import DataSplit
from src.evaluation.fairness import (
    binarise_attribute,
    compute_all_fairness_metrics,
    summarise_fairness,
)
from src.evaluation.mitigation import fit_predict_equalized_odds, fit_predict_prejudice_remover
from src.evaluation.utility import column_correlation_delta, compute_mmd, compute_utility_metrics
from src.models.classifiers import build_classifier
from src.utils.logging import get_logger

logger = get_logger(__name__)


def score_predictions(
    y_test,
    y_pred,
    y_prob,
    protected_test,
    cfg,
    protected_attrs=None
):
    """Compute utility and fairness metrics for one set of test-set predictions."""
    metrics = compute_utility_metrics(y_test, y_pred, y_prob)

    # A classifier that predicts a single class makes DPD and EOD trivially zero, which is
    # indistinguishable from perfect fairness unless it is flagged.
    metrics["degenerate_predictions"] = bool(len(np.unique(y_pred)) < 2)

    fairness_cfg = {**cfg, "fairness_attributes_subset": protected_attrs}
    fairness_df = compute_all_fairness_metrics(y_test, y_pred, protected_test, fairness_cfg)

    for _, row in fairness_df.iterrows():
        for metric in ("dpd", "eod", "di", "odds_ratio"):
            metrics[f"{row['attribute']}_{metric}"] = row[metric]
    metrics.update(summarise_fairness(fairness_df))

    return metrics, fairness_df


class Evaluator:
    """Trains a classifier on a given training set and scores it against the fixed real test set."""

    def __init__(self, split: DataSplit, cfg: dict, protected_attrs=None):
        self.split = split
        self.cfg = cfg
        self.protected_attrs = protected_attrs or split.protected_attrs
        self.last_fairness_detail: pd.DataFrame | None = None

    def baseline(self, classifier_name: str) -> dict:
        logger.info(f"Real data baseline for [{classifier_name}]")
        return self._run(self.split.X_train, self.split.y_train, classifier_name, "real")

    def evaluate(self, X_synth, y_synth, classifier_name: str, generator_name: str):
        logger.info(f"Evaluating [{generator_name} | {classifier_name}]")
        result = self._run(X_synth, y_synth, classifier_name, "synthetic")
        correlation = column_correlation_delta(self.split.X_train, X_synth)

        return {
            "method": generator_name,
            "classifier": classifier_name,
            **result,
            "mmd": compute_mmd(self.split.X_train, X_synth, self.cfg),
            "corr_mean_abs_delta": correlation["mean_abs_delta"],
            "corr_max_abs_delta": correlation["max_abs_delta"],
        }

    def _run(self, X_train, y_train, classifier_name: str, label: str) -> dict:
        clf = build_classifier(classifier_name, self.cfg, y_train=y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(self.split.X_test)
        y_prob = clf.predict_proba(self.split.X_test)[:, 1]

        metrics, fairness_df = score_predictions(
            self.split.y_test.values, y_pred, y_prob,
            self.split.protected_test, self.cfg, self.protected_attrs,
        )
        self.last_fairness_detail = fairness_df

        logger.info(f"  [{label}] AUC={metrics['roc_auc']:.4f} BA={metrics['balanced_accuracy']:.4f}")
        return metrics


class MitigatedEvaluator:
    """Same TSTR setup as Evaluator, with a mitigation technique applied to one protected attribute."""

    def __init__(self, split: DataSplit, cfg: dict, protected_attrs=None):
        self.split = split
        self.cfg = cfg
        self.protected_attrs = protected_attrs or split.protected_attrs

    def evaluate(
        self,
        X_train,
        y_train,
        technique: str,
        target_attr: str,
        method_name: str,
        attr_train: pd.Series,
        classifier_name: str | None = None,
    ) -> dict:
        """attr_train holds the raw protected values of whichever data is being trained on, real or synthetic."""
        self._check_trainable(attr_train, y_train, target_attr, method_name)
        attr_test = self.split.protected_test[target_attr]

        if technique == "equalized_odds":
            clf = build_classifier(classifier_name, self.cfg, y_train=y_train)
            y_pred, y_prob = fit_predict_equalized_odds(
                clf, X_train, y_train, attr_train, self.split.X_test, attr_test, self.cfg,
            )
            model_label = classifier_name
        elif technique == "prejudice_remover":
            y_pred, y_prob = fit_predict_prejudice_remover(
                X_train, y_train, attr_train, self.split.X_test, attr_test, self.cfg,
            )
            model_label = "prejudice_remover"
        else:
            raise ValueError(f"Unknown mitigation technique '{technique}'.")

        metrics, _ = score_predictions(
            self.split.y_test.values, y_pred, y_prob,
            self.split.protected_test, self.cfg, self.protected_attrs,
        )
        logger.info(
            f"  [{technique} | {target_attr} | {method_name} | {model_label}] "
            f"BA={metrics['balanced_accuracy']:.4f} mean_dpd={metrics.get('mean_dpd', float('nan')):.4f}"
        )
        return {
            "technique": technique,
            "target_attr": target_attr,
            "method": method_name,
            "classifier": model_label,
            **metrics,
        }

    def _check_trainable(self, attr_train, y_train, target_attr, method_name):
        """Both libraries need two groups, each carrying both labels; a synthetic set does not always."""
        group = binarise_attribute(attr_train, self.cfg).to_numpy(dtype=float, na_value=np.nan)
        valid = ~np.isnan(group)
        groups = np.unique(group[valid])

        if len(groups) < 2:
            raise ValueError(
                f"Only one group present for '{target_attr}' in the '{method_name}' training data."
            )

        y_valid = np.asarray(y_train)[valid]
        for g in groups:
            if len(np.unique(y_valid[group[valid] == g])) < 2:
                raise ValueError(
                    f"Group {g:.0f} of '{target_attr}' carries a single label in the "
                    f"'{method_name}' training data."
                )
