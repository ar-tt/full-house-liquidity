"""Load config.yaml and fail loudly if something the engine needs is missing."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

REQUIRED_SECTIONS = ["meta", "cash_flows", "decision_dates", "reserve", "projection", "data"]


class ConfigError(ValueError):
    pass


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config.yaml is missing section(s): {', '.join(missing)}")
    cfg["_path"] = str(path)
    return cfg
