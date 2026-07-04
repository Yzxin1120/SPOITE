"""Configuration loading with an auditable content hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    cfg = json.loads(raw)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg
