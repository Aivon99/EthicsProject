from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REQUIREMENTS_PATH = Path(__file__).resolve().parents[2] / "requirements.txt"

_REQUIREMENT_LINE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(==)?\s*([A-Za-z0-9_.\-]*)$")


def check_prerequisites(requirements_path=DEFAULT_REQUIREMENTS_PATH):
    """Verify every package in requirements.txt is installed. Meant to be the first thing a notebook runs."""
    requirements_path = Path(requirements_path)
    missing = []
    mismatched = []
    ok = 0

    for name, expected in _parse_requirements(requirements_path):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if expected and installed != expected:
            mismatched.append(f"{name} installed {installed}, expected {expected}")
        else:
            ok += 1

    print(f"Prerequisites: {ok} satisfied, {len(missing)} missing, {len(mismatched)} mismatched.")
    if missing or mismatched:
        details = "\n".join([f"  missing: {n}" for n in missing] + [f"  {m}" for m in mismatched])
        raise RuntimeError(
            f"Environment does not match {requirements_path.name}:\n{details}\n"
            f"Run: pip install -r {requirements_path}"
        )


def _parse_requirements(requirements_path: Path) -> list[tuple[str, str | None]]:
    requirements = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQUIREMENT_LINE.match(line)
        if not match:
            logger.warning(f"Could not parse requirements line, skipping: {raw_line}")
            continue
        name, _, version = match.groups()
        requirements.append((name, version or None))
    return requirements
