from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.utility import compute_utility_metrics, compute_mmd, column_correlation_delta, utility_delta
from src.evaluation.fairness import compute_all_fairness_metrics, summarise_fairness
from src.evaluation.mitigation import fit_predict_equalized_odds, fit_predict_prejudice_remover
from src.models.classifiers import build_classifier
from src.data.preprocessor import DataSplit
from src.utils import get_logger

logger = get_logger(__name__)


def score_predictions(
    y_test,
    y_pred,
    y_prob,
    protected_test: pd.DataFrame,
    cfg: dict,
    protected_attrs=None,
) -> tuple[dict, pd.DataFrame]:
    """Compute utility + per-attribute/aggregate fairness metrics for one
    set of test-set predictions. Shared by ``Evaluator`` and
    ``MitigatedEvaluator`` so both produce identically-shaped result rows.
    """
    metrics = compute_utility_metrics(y_test, y_pred, y_prob)

    fairness_cfg = {**cfg, "fairness_attributes_subset": protected_attrs}
    fairness_df = compute_all_fairness_metrics(y_test, y_pred, protected_test, fairness_cfg)

    for _, frow in fairness_df.iterrows():
        for m in ("dpd", "eod", "di"):
            metrics[f"{frow['attribute']}_{m}"] = frow[m]
    metrics.update(summarise_fairness(fairness_df))

    return metrics, fairness_df


class Evaluator:
    """Trains a classifier on a given train set and scores it against the
    fixed real test set held by ``split`` (TSTR: train-on-synthetic/real,
    test-on-real).
    """

    def __init__(self, split: DataSplit, cfg: dict, protected_attrs=None):
        self.split = split
        self.cfg = cfg
        self.protected_attrs = protected_attrs or split.protected_attrs
        self._real_metrics: dict | None = None
        # Per-attribute fairness DataFrame from the most recent _run() call,
        # for callers that want the full breakdown (not just the flattened dict).
        self.last_fairness_detail: pd.DataFrame | None = None

    def baseline(self, classifier_name) -> dict:
        logger.info(f"Computing real-data baseline for [{classifier_name}] …")
        real_result = self._run(self.split.X_train, self.split.y_train, classifier_name, "real")
        self._real_metrics = {k: real_result[k] for k in ("balanced_accuracy", "f1_macro", "roc_auc")}
        return real_result

    def evaluate(
        self,
        X_synth: pd.DataFrame,
        y_synth: pd.Series,
        classifier_name,
        generator_name,
        repetition=0,
        X_real_for_mmd=None,
    ) -> dict:
        if self._real_metrics is None:
            self.baseline(classifier_name)

        logger.info(f"Evaluating [{generator_name} | {classifier_name} | rep={repetition}] …")
        synth_result = self._run(X_synth, y_synth, classifier_name, "synthetic")

        synth_clf = {k: synth_result[k] for k in ("balanced_accuracy", "f1_macro", "roc_auc")}
        gap = utility_delta(self._real_metrics, synth_clf)

        X_real = X_real_for_mmd if X_real_for_mmd is not None else self.split.X_train
        mmd_val = compute_mmd(X_real, X_synth, self.cfg)
        corr = column_correlation_delta(X_real, X_synth)

        return {
            "generator": generator_name,
            "classifier": classifier_name,
            "repetition": repetition,
            **synth_result,
            **gap,
            "mmd": mmd_val,
            "corr_mean_abs_delta": corr["mean_abs_delta"],
            "corr_max_abs_delta": corr["max_abs_delta"],
        }

    def _run(self, X_train, y_train, classifier_name, label) -> dict:
        clf = build_classifier(classifier_name, self.cfg, y_train=y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(self.split.X_test)
        y_prob = clf.predict_proba(self.split.X_test)[:, 1]

        metrics, fairness_df = score_predictions(
            self.split.y_test.values, y_pred, y_prob, self.split.protected_test, self.cfg, self.protected_attrs,
        )
        self.last_fairness_detail = fairness_df

        logger.info(f"  [{label}] AUC={metrics['roc_auc']:.4f} | BA={metrics['balanced_accuracy']:.4f}")
        return metrics


class MitigatedEvaluator:
    """Same TSTR pattern as ``Evaluator`` (train on real/synthetic, score
    against the fixed real test set held by ``split``), but training is
    wrapped with a fairness-mitigation technique targeted at ONE protected
    attribute at a time -- the loop over attributes happens in the caller
    (e.g. one attribute per notebook iteration).

    Two techniques (implemented in ``src/evaluation/mitigation.py``):
      - ``"equalized_odds"``:    post-processing wrapper (fairlearn
        ThresholdOptimizer) around any of the 3 classifiers --
        ``classifier_name`` is required.
      - ``"prejudice_remover"``: its own in-processing model (aif360,
        Kamishima et al. 2012) -- ``classifier_name`` is ignored; one model
        per (attribute, method) pair.
    """

    def __init__(self, split: DataSplit, cfg: dict, protected_attrs=None):
        self.split = split
        self.cfg = cfg
        self.protected_attrs = protected_attrs or split.protected_attrs

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        technique: str,
        target_attr: str,
        method_name: str,
        classifier_name: str | None = None,
    ) -> dict:
        attr_train = self.split.protected_train[target_attr]
        attr_test = self.split.protected_test[target_attr]

        if technique == "equalized_odds":
            if classifier_name is None:
                raise ValueError("classifier_name is required for technique='equalized_odds'")
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
            raise ValueError(f"Unknown mitigation technique '{technique}'. Valid options: equalized_odds, prejudice_remover")

        metrics, _ = score_predictions(
            self.split.y_test.values, y_pred, y_prob, self.split.protected_test, self.cfg, self.protected_attrs,
        )
        logger.info(
            f"  [mitigated:{technique} | target={target_attr} | {method_name} | {model_label}] "
            f"BA={metrics['balanced_accuracy']:.4f} | mean_dpd={metrics.get('mean_dpd', float('nan')):.4f}"
        )
        return {
            "technique": technique,
            "target_attr": target_attr,
            "method": method_name,
            "classifier": model_label,
            **metrics,
        }