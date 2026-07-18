from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from src.generation.constraints import add_target_exclusivity_constraint
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_tvae(
    train_df: pd.DataFrame,
    metadata: SingleTableMetadata,
    cfg: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    """Fit TVAE on train_df, sample a synthetic dataset, and save it."""
    method_cfg = cfg["generation"]["methods"]["tvae"]
    n_samples  = _resolve_n_samples(cfg, train_df)
    seed       = cfg["seed"]

    logger.info(
        f"TVAE: fitting on {len(train_df)} rows "
        f"(epochs={method_cfg['epochs']}, "
        f"batch_size={method_cfg['batch_size']}, "
        f"seed={seed})..."
    )

    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=method_cfg["epochs"],
        batch_size=method_cfg["batch_size"],
        enable_gpu=False,
        verbose=True,
    )

    add_target_exclusivity_constraint(synthesizer, cfg)

    synthesizer.fit(train_df)

    logger.info(f"TVAE: sampling {n_samples} synthetic rows...")
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
    logger.info(f"TVAE: synthetic dataset saved -> {path}  ({len(df)} rows)")