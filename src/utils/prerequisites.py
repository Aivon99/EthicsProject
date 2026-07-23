from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REQUIREMENTS_PATH = Path(__file__).resolve().parents[2] / "requirements.txt"

_REQ_LINE_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(==)?\s*([A-Za-z0-9_.\-]*)$")


def _parse_requirements(requirements_path: Path) -> list[tuple[str, str | None]]:
    """Parse a requirements.txt into (package_name, pinned_version_or_None) pairs.

    Skips blank lines and comments. Only understands plain `pkg` / `pkg==x.y.z`
    lines (which is all this project's requirements.txt uses) -- extras,
    environment markers, and VCS/URL requirements are not handled.
    """
    reqs: list[tuple[str, str | None]] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQ_LINE_RE.match(line)
        if not match:
            logger.warning("Could not parse requirements.txt line, skipping: %r", raw_line)
            continue
        name, _, version = match.groups()
        reqs.append((name, version or None))
    return reqs


def check_prerequisites(
    requirements_path: Path | str = DEFAULT_REQUIREMENTS_PATH,
    auto_install: bool = False,
    raise_on_missing: bool = True,
) -> bool:
    """Verify every package in requirements.txt is installed (and, if pinned,
    at the expected version). Meant to be the first thing a notebook runs.

    Parameters
    ----------
    requirements_path : path to requirements.txt (defaults to the project root's).
    auto_install       : if True, run `pip install -r requirements.txt` for any
                          missing/mismatched package instead of just reporting it.
    raise_on_missing   : if True (default) and auto_install is False, raise
                          RuntimeError when something is missing/mismatched so a
                          notebook run fails fast instead of continuing with a
                          broken environment.

    Returns
    -------
    True if every requirement is satisfied (after auto-install, if requested).
    """
    requirements_path = Path(requirements_path)
    if not requirements_path.exists():
        logger.warning("requirements.txt not found at %s -- skipping prerequisite check.", requirements_path)
        return True

    requirements = _parse_requirements(requirements_path)

    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []  # (name, installed, expected)
    ok: list[str] = []

    for name, expected_version in requirements:
        try:
            installed_version = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue

        if expected_version and installed_version != expected_version:
            mismatched.append((name, installed_version, expected_version))
        else:
            ok.append(name)

    print(f"Prerequisites check ({requirements_path.name}): "
          f"{len(ok)} OK, {len(missing)} missing, {len(mismatched)} version-mismatched.")

    if missing:
        print(f"  Missing      : {missing}")
    if mismatched:
        for name, installed, expected in mismatched:
            print(f"  Mismatched   : {name} (installed {installed}, expected {expected})")

    if not missing and not mismatched:
        return True

    if auto_install:
        print(f"  Installing from {requirements_path} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            check=True,
        )
        # Re-check without recursing into auto_install again.
        return check_prerequisites(requirements_path, auto_install=False, raise_on_missing=raise_on_missing)

    if raise_on_missing:
        raise RuntimeError(
            f"{len(missing)} missing / {len(mismatched)} mismatched package(s) "
            f"(see above). Run `pip install -r {requirements_path}` "
            "or call check_prerequisites(auto_install=True)."
        )
    return False
