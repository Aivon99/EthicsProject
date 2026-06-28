from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any



# ---- Public functions --------------------------------------------------------
def load_config(config_path = "../config/config.yaml") -> dict[str, Any]:
    """
    Load the YAML configuration file and relative paths.

    Parameters
    ----------
    config_path : str or Path
        Explicit path to config.yaml.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.  All values under the
        ``paths`` key are ``pathlib.Path`` objects pointing at real
        (or future) filesystem locations.
    """
    resolved_config_path = Path(config_path).resolve()
    project_root = resolved_config_path.parent.parent

    with open(resolved_config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Store the resolved project root for downstream use
    cfg["project_root"] = project_root

    # Resolve all path strings to absolute Path objects
    cfg["paths"] = _resolve_paths(cfg.get("paths", {}), project_root)

    return cfg


def get_synthetic_output_path(cfg: dict, method: str) -> Path:
    """
    Return the full path where synthetic data for a
    given generation method should be written.

    Parameters
    ----------
    cfg : dict
        Configuration dict returned by ``load_config()``.
    method : str
        One of ``ctgan``, ``tvae``, ``gaussian_copula``, ``smote``.

    Returns
    -------
    pathlib.Path
        Full path to the output CSV, e.g.
        ``<project_root>/data/synthetic/ctgan/synthetic_ctgan.csv``.
    """
    subdirs = cfg["generation"]["output_subdirs"]
    filename_template = cfg["generation"]["output_filename_template"]

    subdir = subdirs[method]
    filename = filename_template.format(method=subdir)

    output_path = cfg["paths"]["synthetic_dir"] / subdir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path



# ---- Internal helpers --------------------------------------------------------
def _resolve_paths(paths_section: dict, project_root: Path) -> dict:
    """
    Recursively resolve all string values in the paths section to
    absolute pathlib.Path objects.
    """
    resolved: dict[str, Any] = {}
    for key, value in paths_section.items():
        if isinstance(value, str):
            p = Path(value)
            resolved[key] = p if p.is_absolute() else (project_root / p).resolve()
        elif isinstance(value, dict):
            resolved[key] = _resolve_paths(value, project_root)
        else:
            resolved[key] = value
    return resolved