"""Projection engine: grow the portfolio year by year up to the carve-out.

It takes a return for each calendar year. The deterministic case passes the
same rate every year; the Monte Carlo (step 5) will pass thousands of random
paths through this same function.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Union

from .cashflows import Schedule
from .config import ConfigError

Returns = Union[float, dict, Callable[[int], float]]


@dataclass
class YearRow:
    year: int
    value_start: float     # value on Jan 1, before that day's cash flows
    flows: float           # cash in (+) or out (-) on Jan 1
    value_invested: float  # value after the flows, invested for the year
    rate: float
    growth: float
    value_end: float       # value on Dec 31 (= next Jan 1)


@dataclass
class Projection:
    rows: list[YearRow]
    end_date: date

    @property
    def final_value(self) -> float:
        return self.rows[-1].value_end if self.rows else 0.0


def _rate_fn(returns: Returns) -> Callable[[int], float]:
    if callable(returns):
        return returns
    if isinstance(returns, dict):
        def lookup(year: int) -> float:
            if year not in returns:
                raise ConfigError(f"no return given for {year}")
            return returns[year]
        return lookup
    return lambda _year: float(returns)


def project(schedule: Schedule, until: date, returns: Returns) -> Projection:
    """Portfolio value on `until` (a Jan 1), applying every cash flow dated
    before it. Flows on or after `until` belong to the reserve, not here."""
    if (until.month, until.day) != (1, 1):
        raise ConfigError(f"projection end date must be a Jan 1, got {until}")
    flows = schedule.before(until)
    if not flows:
        raise ConfigError(f"no cash flows before {until}; nothing to project")
    odd = [f for f in flows if (f.date.month, f.date.day) != (1, 1)]
    if odd:
        raise ConfigError("the annual projection only handles start-of-year (Jan 1) cash flows; "
                          f"found {[str(f.date) for f in odd]}")
    rate_for = _rate_fn(returns)
    by_year: dict[int, float] = {}
    for f in flows:
        by_year[f.date.year] = by_year.get(f.date.year, 0.0) + f.amount

    rows, value = [], 0.0
    for year in range(flows[0].date.year, until.year):
        cash = by_year.get(year, 0.0)
        invested = value + cash
        if invested < 0:
            raise ConfigError(f"portfolio goes negative in {year}: outflows exceed assets")
        r = rate_for(year)
        end = invested * (1 + r)
        rows.append(YearRow(year, value, cash, invested, r, end - invested, end))
        value = end
    return Projection(rows, until)


@dataclass
class FacilityResult:
    portfolio_value: float
    reserve_cost: float

    @property
    def fully_funded(self) -> bool:
        return self.portfolio_value >= self.reserve_cost

    @property
    def contribution(self) -> float:
        """The reserve is funded first; the facility only gets what is left."""
        return max(0.0, self.portfolio_value - self.reserve_cost)

    @property
    def shortfall(self) -> float:
        return max(0.0, self.reserve_cost - self.portfolio_value)


def facility_contribution(portfolio_value: float, reserve_cost: float) -> FacilityResult:
    return FacilityResult(portfolio_value, reserve_cost)
