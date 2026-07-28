from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path="../config/config.yaml") -> dict:
    """Load config file."""
    resolved_config_path = Path(config_path).resolve()
    project_root = resolved_config_path.parent.parent

    with open(resolved_config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["project_root"] = project_root
    cfg["paths"] = _resolve_paths(cfg.get("paths", {}), project_root)
    return cfg


def select_task(cfg: dict, task: str) -> dict:
    """Return a copy of cfg bound to one task: either 'target_high_perf' or 'target_low_perf'."""
    if task not in cfg["tasks"]:
        raise ValueError(f"Unknown task '{task}'. Valid options: {list(cfg['tasks'])}")

    task_cfg = copy.deepcopy(cfg)
    task_cfg["task"] = task
    task_cfg["dataset"]["target_column"] = cfg["tasks"][task]["column"]
    task_cfg["generation"]["datasets"] = cfg["generation"]["datasets_per_task"][task]
    return task_cfg


def get_synthetic_output_path(cfg: dict, method: str) -> Path:
    """Return the output CSV path for a given generation method."""
    subdir = cfg["generation"]["output_subdirs"][method]
    filename = cfg["generation"]["output_filename_template"].format(method=subdir)

    output_path = cfg["paths"]["synthetic_dir"] / subdir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def notebook_dirs(cfg: dict, notebook: str) -> tuple[Path, Path]:
    """Return and create the results and figures directories belonging to one notebook."""
    results_dir = cfg["paths"]["results_dir"] / notebook
    figures_dir = cfg["paths"]["figures_dir"] / notebook
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, figures_dir


def _resolve_paths(paths_section: dict, project_root: Path) -> dict:
    resolved: dict[str, Any] = {}
    for key, value in paths_section.items():
        # A URL is not a path and must survive untouched.
        if key.endswith("_url") or not isinstance(value, str) or not value:
            resolved[key] = value
        else:
            path = Path(value)
            resolved[key] = path if path.is_absolute() else (project_root / path).resolve()
    return resolved
