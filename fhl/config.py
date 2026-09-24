"""Load config.yaml and fail loudly if something the engine needs is missing."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
LOCAL_NAME = "config.local.yaml"

REQUIRED_SECTIONS = ["meta", "cash_flows", "decision_dates", "reserve", "projection", "data"]


class ConfigError(ValueError):
    pass


def _merge(base: dict, extra: dict) -> dict:
    """Settings in `extra` replace the same settings in `base`, key by key."""
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml, plus config.local.yaml beside it if present. The local
    file is for private settings (e.g. the SEC contact email) and is kept out
    of git, so it never reaches the public repository."""
    path = Path(path) if path else DEFAULT_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    local = path.with_name(LOCAL_NAME)
    if local.exists():
        with open(local) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config.yaml is missing section(s): {', '.join(missing)}")
    cfg["_path"] = str(path)
    return cfg


def with_changes(cfg: dict, changes: dict, prefix: str = "") -> dict:
    """A copy of the config with some settings changed, for what-ifs. Keys are
    dotted names, e.g. {"wind_down.hedge_ramp": [...]} (under `prefix` if given).
    Unknown names are refused, so a typo can't silently do nothing."""
    import copy
    out = copy.deepcopy(cfg)
    for path, value in changes.items():
        full = f"{prefix}.{path}" if prefix else path
        node, keys = out, full.split(".")
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise ConfigError(f"unknown setting '{full}'")
            node = node[k]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise ConfigError(f"unknown setting '{full}'")
        node[keys[-1]] = value
    return out
