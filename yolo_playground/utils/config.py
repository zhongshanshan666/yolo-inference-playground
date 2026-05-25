from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML config and merge with optional overrides."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    default_path = Path(path).parent / "default.yaml"
    if default_path.exists() and Path(path).name != "default.yaml":
        with open(default_path, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = _deep_merge(base, cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
