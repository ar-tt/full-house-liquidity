"""Holdings, proposed trades, and exposures (how much money sits in each bucket)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .config import ConfigError
from .securities import Security


@dataclass(frozen=True)
class Proposal:
    """A trade someone wants to make. Buys are paid from cash; sells go to cash."""
    ticker: str
    action: str       # buy | sell
    amount: float     # dollars
    date: date

    def __post_init__(self):
        if self.action not in ("buy", "sell"):
            raise ConfigError(f"action must be buy or sell, got '{self.action}'")
        if self.amount <= 0:
            raise ConfigError("trade amount must be positive")


@dataclass
class Portfolio:
    holdings: dict = field(default_factory=dict)   # ticker -> market value ($)
    cash: float = 0.0

    @classmethod
    def from_file(cls, path: str | Path) -> "Portfolio":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        h = {str(k).upper(): float(v) for k, v in (raw.get("holdings") or {}).items()}
        return cls(h, float(raw.get("cash", 0)))

    @property
    def total(self) -> float:
        return sum(self.holdings.values()) + self.cash

    def weight(self, ticker: str) -> float:
        return self.holdings.get(ticker, 0.0) / self.total if self.total else 0.0

    def after(self, p: Proposal) -> "Portfolio":
        h = dict(self.holdings)
        if p.action == "buy":
            h[p.ticker] = h.get(p.ticker, 0.0) + p.amount
            cash = self.cash - p.amount
        else:
            left = h.get(p.ticker, 0.0) - p.amount
            if left < -1e-6:
                raise ConfigError(f"can't sell ${p.amount:,.0f} of {p.ticker}: only "
                                  f"${h.get(p.ticker, 0):,.0f} held")
            if left <= 1e-6:
                h.pop(p.ticker, None)
            else:
                h[p.ticker] = left
            cash = self.cash + p.amount
        return Portfolio(h, cash)

    def tickers(self, securities: dict, asset_classes: list | None = None) -> list[str]:
        out = []
        for t in self.holdings:
            if t not in securities:
                raise ConfigError(f"{t} is held but missing from securities.yaml")
            if asset_classes is None or securities[t].asset_class in asset_classes:
                out.append(t)
        return sorted(out)


def exposures(pf: Portfolio, securities: dict[str, Security], dimension: str,
              asset_classes: list | None = None) -> dict[str, float]:
    """Share of the portfolio in each bucket of `dimension`
    (asset_class | sector | region | issuer), ETFs split by look-through."""
    out: dict[str, float] = {}
    if dimension == "asset_class" and pf.cash and (asset_classes is None or "cash" in asset_classes):
        out["cash"] = pf.cash / pf.total
    for t, value in pf.holdings.items():
        s = securities[t]
        if asset_classes is not None and s.asset_class not in asset_classes:
            continue
        w = value / pf.total
        if dimension == "asset_class":
            parts = {s.asset_class: 1.0}
        elif dimension == "sector":
            parts = s.sector_weights
        elif dimension == "region":
            parts = s.region_weights
        elif dimension == "issuer":
            parts = {s.issuer: 1.0}
        else:
            raise ValueError(dimension)
        for k, share in parts.items():
            out[k] = out.get(k, 0.0) + w * share
    return out
