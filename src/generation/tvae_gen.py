from __future__ import annotations

from typing import Any

import pandas as pd
from sdv.single_table import TVAESynthesizer

from src.generation.constraints import add_target_exclusivity_constraint
from src.generation.io import resolve_n_samples, save_synthetic, seed_generator
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_tvae(
    train_df: pd.DataFrame,
    metadata,
    cfg: dict,
    output_path
) -> pd.DataFrame:
    """Fit TVAE on the training set, sample a synthetic dataset and save it."""
    seed = seed_generator(cfg)
    params = cfg["generation"]["methods"]["tvae"]
    n_samples = resolve_n_samples(cfg, train_df)

    logger.info(
        f"TVAE: fitting on {len(train_df)} rows "
        f"(epochs={params['epochs']}, batch_size={params['batch_size']}, seed={seed})"
    )

    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        enable_gpu=False,
        verbose=True,
    )
    add_target_exclusivity_constraint(synthesizer, cfg)
    synthesizer.fit(train_df)

    logger.info(f"TVAE: sampling {n_samples} rows")
    synthetic_df = synthesizer.sample(num_rows=n_samples)

    save_synthetic(synthetic_df, output_path, "TVAE")
    return synthetic_df
