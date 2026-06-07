from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any




_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "config.yaml"




def load_config(path = None):
    config_path = Path(path) if path else _DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError("config file not found")

    with open(config_path, "r") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = _PROJECT_ROOT / value

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