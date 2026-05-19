from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml


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


def get_project_root():
    return _PROJECT_ROOT