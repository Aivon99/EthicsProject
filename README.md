# Fair Synthetic Data Generation for Educational Performance Prediction

Project for the *AI Ethics* course (University of Bologna). We study whether
synthetic data generation (CTGAN, TVAE, Gaussian Copula, SMOTE) can be used
to train performance-prediction models that are as accurate and as fair as
models trained on real student data, and whether bias-mitigation techniques
can further close any fairness gap.

The dataset is the *Unfair Inequality in Education* benchmark (Aequitas
collection, [Zenodo record](https://zenodo.org/records/17592007)), containing
student questionnaire and performance data annotated with protected
attributes (gender, birth country, parental education/occupation,
socioeconomic composite, school type, etc.).

## Goals

1. **Preprocessing** — clean the raw dataset, engineer a binary
   high/low-performance target from the maths score, and select a reduced,
   non-sensitive feature set.
2. **Synthetic generation** — generate synthetic training data with CTGAN,
   TVAE, Gaussian Copula and SMOTE, and check fidelity against the real
   distribution.
3. **TSTR (Train-Synthetic-Test-Real) experiments** — train classifiers
   (logistic regression, XGBoost, MLP) on real vs. synthetic data and compare
   utility and fairness metrics.
4. **Mitigation** — apply fairness-mitigation techniques (equalized odds,
   prejudice remover) and measure their effect on utility/fairness trade-offs.
5. **Cross-comparison** — aggregate results across generation methods,
   classifiers and mitigation strategies.

## Repository structure

```
config/         config.yaml — single source of truth for paths, seed, target
                definition, protected attributes, generation/model
                hyperparameters, and experiment settings
notebook/       pipeline notebooks, run in order:
                G1Preprocessing.ipynb
                G2_Synthetic_Generation.ipynb
                G3.ipynb / G3_LowPerf.ipynb
                G4_TSTR_Experiments_No_Enforcing.ipynb / G4_LowPerf_...
                G5_Cross_Comparison.ipynb
src/
  data/         loading, preprocessing, imputation, feature selection
  generation/   CTGAN/TVAE/Copula/SMOTE generators, constraints, fidelity checks
  evaluation/   utility, fairness, mitigation and delta metrics
  models/       classifier wrappers
  utils/        config loading, logging, plotting, prerequisite checks
Data/           local working copies of raw/processed/train/test data
                (gitignored — see Setup below)
results/        generated figures, synthetic datasets, experiment outputs
                (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

`Data/` is not tracked in git. `src/utils/prerequisites.py` / the first
notebook cell will download the raw dataset automatically from the Google
Drive URL configured in `config/config.yaml` (`paths.raw_data_url`) if
`Data/original.csv` is not found locally.

## Running the pipeline

Run the notebooks in `notebook/` in numeric order (G1 → G5). Each notebook
reads its configuration from `config/config.yaml` and writes its outputs
under `results/`, which are consumed by the following notebook.

## Notes

- `students-dataset/` is a bundled copy of the external Aequitas benchmark
  repository (raw data + reference preprocessing/benchmark code), kept for
  reference only and excluded from git — it is not part of the work produced
  for this project.
- Reproducibility seed and all experiment/model hyperparameters are
  centralized in `config/config.yaml`.
