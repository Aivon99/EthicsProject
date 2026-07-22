from __future__ import annotations

from typing import Any

from sdv.cag import FixedCombinations


def add_target_exclusivity_constraint(synthesizer, cfg: dict[str, Any]) -> None:
    """Forbid target_high_perf and target_low_perf from both being 1, since they are mutually exclusive by construction."""
    high_col = cfg["target"]["column_name"]
    low_col = cfg["target"]["low_column_name"]
    synthesizer.add_constraints(constraints=[
        FixedCombinations(column_names=[high_col, low_col])
    ])
