"""Security master: what each ticker IS (type, asset class, sector, region).

A stock has one sector and one region. An ETF can spread across several:
give `sector_weights` / `region_weights` (shares that add up to 1) and the
Range Table counts the ETF's money in each bucket pro rata ("look-through").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import ConfigError

TYPES = ("stock", "etf", "bond")


@dataclass(frozen=True)
class Security:
    ticker: str
    name: str
    type: str                 # stock | etf | bond
    asset_class: str          # equity | bond | cash (from config)
    issuer: str
    sector_weights: dict = field(default_factory=dict)
    region_weights: dict = field(default_factory=dict)
    has_prices: bool = True   # False for things with no daily price feed (e.g. STRIPS)
    cik: int | None = None    # SEC company number, for fundamentals

    @property
    def main_sector(self) -> str:
        return max(self.sector_weights, key=self.sector_weights.get)


def _weights(raw: dict, single_key: str, multi_key: str, ticker: str, allowed: list) -> dict:
    if multi_key in raw:
        w = {k: float(v) for k, v in raw[multi_key].items()}
    elif single_key in raw:
        w = {raw[single_key]: 1.0}
    else:
        raise ConfigError(f"{ticker}: needs '{single_key}' or '{multi_key}'")
    bad = [k for k in w if k not in allowed]
    if bad:
        raise ConfigError(f"{ticker}: unknown {single_key} {bad}; allowed: {allowed}")
    total = sum(w.values())
    if abs(total - 1) > 1e-6:
        raise ConfigError(f"{ticker}: {multi_key} add up to {total:.4f}, must be 1")
    return w


def load_securities(cfg: dict, path: str | Path | None = None) -> dict[str, Security]:
    rt = cfg["range_table"]
    path = Path(path) if path else Path(cfg["_path"]).parent / cfg["securities_file"]
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    out = {}
    for ticker, s in (raw.get("securities") or {}).items():
        ticker = str(ticker).upper()
        if s.get("type") not in TYPES:
            raise ConfigError(f"{ticker}: type must be one of {TYPES}")
        if s.get("asset_class") not in rt["asset_classes"]:
            raise ConfigError(f"{ticker}: asset_class must be one of {rt['asset_classes']}")
        out[ticker] = Security(
            ticker=ticker,
            name=s.get("name", ticker),
            type=s["type"],
            asset_class=s["asset_class"],
            issuer=s.get("issuer", s.get("name", ticker)),
            sector_weights=_weights(s, "sector", "sector_weights", ticker, rt["sectors"]),
            region_weights=_weights(s, "region", "region_weights", ticker, rt["regions"]),
            has_prices=s.get("has_prices", True),
            cik=int(s["cik"]) if s.get("cik") else None,
        )
    return out
