from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any



def load_config(config_path: str | Path = "../config/config.yaml") -> dict[str, Any]:
    """Load config.yaml and resolve every value under `paths` to an absolute Path."""
    resolved_config_path = Path(config_path).resolve()
    project_root = resolved_config_path.parent.parent

    with open(resolved_config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["project_root"] = project_root
    cfg["paths"] = _resolve_paths(cfg.get("paths", {}), project_root)

    return cfg


def get_synthetic_output_path(cfg: dict, method: str) -> Path:
    """Return the output CSV path for a given generation method, creating its directory."""
    subdirs = cfg["generation"]["output_subdirs"]
    filename_template = cfg["generation"]["output_filename_template"]

    subdir = subdirs[method]
    filename = filename_template.format(method=subdir)

    output_path = cfg["paths"]["synthetic_dir"] / subdir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _resolve_paths(paths_section: dict, project_root: Path) -> dict:
    """Recursively resolve string path values to absolute Path objects."""
    resolved: dict[str, Any] = {}
    for key, value in paths_section.items():
        if isinstance(value, str):
            if value == "":
                # Empty string means "unset" (e.g. blank raw_data_url); keep it
                # as-is so it stays falsy rather than resolving to project_root.
                resolved[key] = value
            else:
                p = Path(value)
                resolved[key] = p if p.is_absolute() else (project_root / p).resolve()
        elif isinstance(value, dict):
            resolved[key] = _resolve_paths(value, project_root)
        else:
            resolved[key] = value
    return resolved