from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

from src.generation.constraints import add_target_exclusivity_constraint
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_gaussian_copula(
    train_df: pd.DataFrame,
    metadata: SingleTableMetadata,
    cfg: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    """Fit a Gaussian Copula on train_df, sample a synthetic dataset, and save it."""
    n_samples = _resolve_n_samples(cfg, train_df)

    logger.info(f"GaussianCopula: fitting on {len(train_df)} rows...")

    synthesizer = GaussianCopulaSynthesizer(
        metadata=metadata
    )

    add_target_exclusivity_constraint(synthesizer, cfg)

    synthesizer.fit(train_df)

    logger.info(f"GaussianCopula: sampling {n_samples} synthetic rows...")
    synthetic_df = synthesizer.sample(num_rows=n_samples)

    _save(synthetic_df, output_path)
    return synthetic_df


def _resolve_n_samples(cfg: dict, train_df: pd.DataFrame) -> int:
    n = cfg["generation"].get("n_synthetic_samples")
    if n is None:
        return len(train_df)
    return int(n)


def _save(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"GaussianCopula: synthetic dataset saved -> {path}  ({len(df)} rows)")