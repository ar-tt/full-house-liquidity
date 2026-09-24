"""The common interface every pillar follows.

A pillar looks at one proposed trade against the current portfolio and
returns a PillarResult: a score from 0 to 100 (or None for "no opinion"),
plus every rule it checked, whether it passed, and why. A rule with
severity "block" that fails stops the order.

Adding a pillar = one new file with a Pillar subclass + one line under
`pillars:` in config.yaml. Nothing else changes.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import date

from ..config import ConfigError
from ..portfolio import Portfolio, Proposal
from ..prices import PriceSource

BLOCK, WAIT, WARN, INFO = "block", "wait", "warn", "info"   # wait = "not yet": the order is held, not refused


@dataclass
class Rule:
    code: str              # short id, e.g. "sector_cap"
    passed: bool
    message: str           # plain English, shown by --explain and in the decision log
    severity: str = INFO   # block | wait | warn | info (what happens if it fails)
    value: float | None = None
    limit: float | None = None

    @property
    def fired(self) -> bool:
        return not self.passed


@dataclass
class PillarResult:
    pillar: str
    score: float | None
    rules: list[Rule] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return any(r.fired and r.severity == BLOCK for r in self.rules)

    @property
    def waiting(self) -> bool:
        return any(r.fired and r.severity == WAIT for r in self.rules)

    @property
    def fired(self) -> list[Rule]:
        return [r for r in self.rules if r.fired]


@dataclass
class Context:
    """Everything a pillar may read. `as_of` makes it point-in-time: pillars
    must never look at data dated after it."""
    cfg: dict
    securities: dict
    prices: PriceSource
    as_of: date
    fundamentals: object = None   # FundamentalsSource (company filings)
    curve: object = None          # Treasury curve as of as_of (for the risk-free rate)


class Pillar:
    name = "pillar"

    def __init__(self, spec: dict, ctx: Context):
        self.spec = spec
        self.ctx = ctx
        self.name = spec["name"]
        self.weight = float(spec["weight"])
        self.structural = bool(spec.get("structural", False))

    def evaluate(self, proposal: Proposal, portfolio: Portfolio) -> PillarResult:
        raise NotImplementedError


def load_pillars(cfg: dict, ctx: Context) -> list[Pillar]:
    """Build every enabled pillar from config, structural ones first."""
    out = []
    for spec in cfg["pillars"]:
        for key in ("name", "module", "weight"):
            if key not in spec:
                raise ConfigError(f"pillar entry {spec} is missing '{key}'")
        if not spec.get("enabled", True):
            continue
        mod_name, _, cls_name = spec["module"].partition(":")
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except (ImportError, AttributeError) as e:
            raise ConfigError(f"pillar '{spec['name']}': can't load {spec['module']} ({e})") from e
        if not issubclass(cls, Pillar):
            raise ConfigError(f"pillar '{spec['name']}': {spec['module']} is not a Pillar")
        out.append(cls(spec, ctx))
    if not out:
        raise ConfigError("no pillars enabled in config.yaml")
    return sorted(out, key=lambda p: not p.structural)  # stable: config order kept within groups
