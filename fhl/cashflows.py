"""The liability model: Laura's cash-flow schedule as a list of rows.

Nothing here assumes how many contributions or payments there are, or when.
Everything comes from the `cash_flows` rows in config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .config import ConfigError


@dataclass(frozen=True)
class CashFlow:
    date: date
    amount: float  # > 0 into the portfolio, < 0 out of it
    kind: str
    note: str = ""

    @property
    def is_inflow(self) -> bool:
        return self.amount > 0


class Schedule:
    def __init__(self, rows: list[CashFlow]):
        self.rows = sorted(rows, key=lambda r: r.date)

    @classmethod
    def from_config(cls, cfg: dict) -> "Schedule":
        rows = []
        for i, raw in enumerate(cfg["cash_flows"], start=1):
            for field in ("date", "amount", "kind"):
                if field not in raw:
                    raise ConfigError(f"cash_flows row {i} is missing '{field}': {raw}")
            if not isinstance(raw["date"], date):
                raise ConfigError(f"cash_flows row {i}: date must be YYYY-MM-DD, got {raw['date']!r}")
            if raw["amount"] == 0:
                raise ConfigError(f"cash_flows row {i}: amount is zero, remove the row instead")
            rows.append(CashFlow(raw["date"], float(raw["amount"]), raw["kind"], raw.get("note", "")))
        return cls(rows)

    def of_kind(self, *kinds: str) -> list[CashFlow]:
        return [r for r in self.rows if r.kind in kinds]

    def contributions(self) -> list[CashFlow]:
        return [r for r in self.rows if r.is_inflow]

    def outflows(self) -> list[CashFlow]:
        return [r for r in self.rows if not r.is_inflow]

    def total_contributed(self) -> float:
        return sum(r.amount for r in self.contributions())

    def total_paid_out(self) -> float:
        return -sum(r.amount for r in self.outflows())

    def before(self, d: date) -> list[CashFlow]:
        """Rows strictly before date d."""
        return [r for r in self.rows if r.date < d]

    def on_or_after(self, d: date) -> list[CashFlow]:
        return [r for r in self.rows if r.date >= d]
