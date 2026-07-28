from __future__ import annotations

from typing import Any

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer

from src.generation.constraints import add_target_exclusivity_constraint
from src.generation.io import resolve_n_samples, save_synthetic
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_gaussian_copula(
    train_df: pd.DataFrame,
    metadata, cfg: dict,
    output_path
) -> pd.DataFrame:
    """Fit a Gaussian copula on the training set, sample a synthetic dataset and save it."""
    n_samples = resolve_n_samples(cfg, train_df)

    logger.info(f"GaussianCopula: fitting on {len(train_df)} rows")
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    add_target_exclusivity_constraint(synthesizer, cfg)
    synthesizer.fit(train_df)

    logger.info(f"GaussianCopula: sampling {n_samples} rows")
    synthetic_df = synthesizer.sample(num_rows=n_samples)

    save_synthetic(synthetic_df, output_path, "GaussianCopula")
    return synthetic_df
