"""Configuration loading with an auditable content hash."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    cfg = json.loads(raw)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def current_git_commit(project_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable-no-project-git-repository"
