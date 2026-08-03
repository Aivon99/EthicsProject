# Fairness and Utility Under Synthetic Data: Measuring what we lose when we anonymise

> Exam project for the course "Ethics in Artificial Intelligence", Master's Degree in Artificial Intelligence, University of Bologna, a.y. 2025-2026.

**Authors**:
- Ivo Rambaldi [[_Github profile_](https://github.com/Aivon99)][[_Institutional_ _email_](mailto:ivo.rambaldi@studio.unibo.it)]
- Petrelli Tommaso [[_Github profile_](https://github.com/petrello)][[_Institutional_ _email_](mailto:tommaso.petrelli2@studio.unibo.it)]

## Overview

Synthetic data is adopted as a privacy preserving substitute for releasing real records. The
substitution is not free: generation distorts the distribution, and the distortion costs both
predictive accuracy and, less visibly, fairness.

This project measures that cost on a real educational dataset, under a **Train on Synthetic, Test
on Real** protocol. We fit four generators (CTGAN, TVAE, Gaussian copula, SMOTE) on the real
training set, train three classifiers (logistic regression, XGBoost, MLP) on each synthetic
replacement, and score every model on the same held-out real students. The whole grid is then run
twice, once predicting the top quarter of the score distribution and once the bottom quarter.

The five questions the notebooks answer, in order:

1. How much accuracy is lost when a classifier is trained on synthetic data instead of real data.
2. How much of the fairness profile changes.
3. Whether the two losses are related.
4. Whether the answers hold when the predicted tail is mirrored from the top quarter to the bottom.
5. Whether standard fairness mitigation still works once the training data is synthetic.

**Results and discussion are in the report.** `notebook/G5_Cross_Comparison.ipynb` reproduces the
same numbers and figures from the outputs of the earlier notebooks.

## Dataset

The Aequitas benchmark ["Unfair Inequality in Education"](https://zenodo.org/records/17592007),
curated by Giovanelli et al.: 83857 students and 561 columns covering the Canary Islands school
system between academic years 2015/16 and 2018/19, from student, family, principal and teacher questionnaires, plus standardised test scores. Data is provided by the Canary Agency
for University Quality and Educational Evaluation (ACCUEE).

## Goals

1. **Preprocessing (G1).** Clean the raw survey data, derive the two prediction targets from the
   maths score (top quartile for excellence, bottom quartile for underperformance) and reduce
   the surviving columns to the 25 most informative, keeping every protected attribute.
2. **Synthetic generation (G2).** Replace the real training set with one synthetic copy per
   technique (CTGAN, TVAE, Gaussian copula, SMOTE) and measure how far each copy drifts from the
   real distribution.
3. **TSTR evaluation (G3).** Train logistic regression, XGBoost and MLP on the real set and on
   each synthetic one, score them all on the same real test students, and read the utility and
   fairness cost of the substitution off the difference.
4. **Mitigation (G4).** Apply equalized odds and prejudice remover to that grid and check whether
   the correction works as well on synthetic training data as it does on real.
5. **Cross-comparison (G5).** Aggregate the four runs to see whether the utility loss and the
   fairness change move together, and whether either conclusion survives flipping the task from
   excellence to underperformance.

## What this repository contains

The experiment lives in seven notebooks that run in order, each reading what the previous one
wrote to disk. `src/` holds the implementation they call; the notebooks themselves are narrative
plus orchestration.

| Notebook | What it does |
| --- | --- |
| [G1_Preprocessing](notebook/G1_Preprocessing.ipynb) | Cleaning, target definition, train/test split, feature selection |
| [G2_Synthetic_Generation](notebook/G2_Synthetic_Generation.ipynb) | Fits the four generators and measures fidelity |
| [G3_TSTR_Evaluation](notebook/G3_TSTR_Evaluation.ipynb) | The TSTR grid, excellence task |
| [G3_TSTR_Evaluation_LowPerf](notebook/G3_TSTR_Evaluation_LowPerf.ipynb) | The same grid, underperformance task |
| [G4_Fairness_Mitigation](notebook/G4_Fairness_Mitigation.ipynb) | Mitigation on top of G3, excellence task |
| [G4_Fairness_Mitigation_LowPerf](notebook/G4_Fairness_Mitigation_LowPerf.ipynb) | The same, underperformance task |
| [G5_Cross_Comparison](notebook/G5_Cross_Comparison.ipynb) | Aggregates the four runs, draws the conclusions |

Two conventions are worth knowing before reading the code. G3 owns the unmitigated grid and writes it to disk; G4 loads that file rather than refitting, so the whole project rests on one baseline. And each task notebook and its `LowPerf` twin differ in exactly one code cell, the one that sets `TASK`.

**Repository structure**

```
config/config.yaml    single source of truth: paths, seed, target definitions,
                      protected attributes, generation and model hyperparameters
src/
  data/               loading, preprocessing, imputation, feature selection
  generation/         CTGAN, TVAE, copula, SMOTE, plus constraints and fidelity
  evaluation/         utility, fairness, mitigation, the delta matrix
  models/             the three classifiers
  utils/              config, logging, plotting, prerequisite checks
Data/                 raw and split data (not tracked)
results/              synthetic datasets, figures, experiment outputs (not tracked)
```

## How to use it

Install the dependencies in a virtual environment:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # Windows
.venv/bin/pip install -r requirements.txt         # macOS / Linux
```

Then run the notebooks in `notebook/` in numeric order, G1 through G5. **The order matters**: each one reads files the previous one wrote. Every notebook's first cell verifies that the installed packages match `requirements.txt` and stops with the list of what is missing if they do not.

A few things to expect:

- **Runtime.** A full run takes about six hours.
- **Configuration.** Change the seed, the task thresholds, the protected attributes or any
  hyperparameter in `config/config.yaml` rather than in the notebooks.
- **Reproducibility.** Every stage is deterministic under the single seed in `config.yaml`, so
  re-running from the raw data regenerates the same synthetic datasets, figures and tables as
  those reported.
