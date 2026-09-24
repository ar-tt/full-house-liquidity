"""BACKTEST: Laura's plan replayed through real history.

Every month from 1981 to 2020 is tried as "day one". From it, the plan runs
six years exactly as the live engine would run it: same wind-down rule, same
risk dial, same risk controls, same Treasury ladder. The difference from the
Monte Carlo is that nothing is invented:

  stocks   the S&P 500 index fund's actual monthly total returns
  ladder   priced on the Fed's actual Treasury curve on each date
  cash     earns the actual 3-month Treasury rate on each date

Each window is one "path", so the same simulation code runs it. The
calendar is kept in plan time (2027-2033): month k of every window is
month k of Laura's plan, fed with what happened in month k of that window.

Two checks come out of it:
  1. CERTAINTY: in how many real histories were all ten payments secured?
  2. QUOTE CHECK: in each window, the 2031-style quote is made using only
     what was known then (the portfolio value and interest rates on that
     day, plus the plan's assumptions). Then we see where the facility money
     actually landed. A well-calibrated quote lands below the floor about
     20% of the time and above the ceiling about 30% of the time.

Windows overlap (a crash appears in many of them), so 476 windows are
roughly seven independent six-year stretches of history, not 476.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from .cashflows import Schedule
from .config import ConfigError, with_changes
from .curves import year_fraction
from .history import CurveHistory, flat_equivalent, index_returns, month_key, zero_rates
from .montecarlo import Market, SimResult, make_market, month_starts, simulate


def add_months(d: date, k: int) -> date:
    m = d.month - 1 + k
    return date(d.year + m // 12, m % 12 + 1, 1)


def window_starts(cfg: dict, returns: dict, curves: CurveHistory, months: int) -> list[date]:
    first = cfg["backtest"]["first_start"]
    look = cfg["risk_controls"]["vol_lookback_months"]
    out, s = [], date(first.year, first.month, 1)
    while add_months(s, months) <= curves.dates[-1]:
        needed = [month_key(add_months(s, k)) for k in range(-look, months)]
        if all(k in returns for k in needed):      # skip windows the data can't fully cover
            out.append(s)
        s = add_months(s, 1)
    if not out:
        raise ConfigError("no complete historical window: check backtest.first_start and the data")
    return out


def build_market(cfg: dict, schedule: Schedule, starts: list[date], returns: dict,
                 curves: CurveHistory) -> Market:
    plan = month_starts(cfg["monte_carlo"]["start"], cfg["decision_dates"]["reserve_carve_out"])
    months = len(plan) - 1
    look = cfg["risk_controls"]["vol_lookback_months"]
    pays = schedule.of_kind(*cfg["reserve"]["funds_kinds"])
    growth = np.array([[returns[month_key(add_months(s, k))] for k in range(months)] for s in starts])
    prior = np.array([[returns[month_key(add_months(s, k))] for k in range(-look, 0)] for s in starts])
    rates = np.empty((len(starts), months + 1))
    bills = np.empty((len(starts), months))
    bill_tau = np.array([cfg["backtest"]["bill_maturity_years"]])
    for k, d in enumerate(plan):
        params = curves.params_on([add_months(s, k) for s in starts])
        left = [p for p in pays if p.date >= d]
        taus = np.array([year_fraction(d, p.date) for p in left])
        rates[:, k] = flat_equivalent(params, taus, np.array([-p.amount for p in left]))
        if k < months:
            bills[:, k] = np.expm1(zero_rates(params, bill_tau)[:, 0] / 12)
    vol = float(np.log1p(np.concatenate([growth.ravel(), prior.ravel()])).std(ddof=1) * math.sqrt(12))
    return Market(growth, np.zeros_like(growth), vol, rates=rates, bills=bills, prior=prior)


@dataclass
class Backtest:
    starts: list
    market: Market
    result: SimResult
    floor: np.ndarray | None = None
    ceiling: np.ndarray | None = None

    @property
    def ends(self) -> list:
        return [add_months(s, len(self.result.dates) - 1) for s in self.starts]


def load(cfg: dict, schedule: Schedule, offline: bool = False) -> tuple[list, Market]:
    returns = index_returns(cfg, offline)
    curves = CurveHistory(cfg, offline)
    months = len(month_starts(cfg["monte_carlo"]["start"], cfg["decision_dates"]["reserve_carve_out"])) - 1
    starts = window_starts(cfg, returns, curves, months)
    return starts, build_market(cfg, schedule, starts, returns, curves)


def replay(cfg: dict, schedule: Schedule, starts: list, market: Market) -> Backtest:
    m = cfg["monte_carlo"]
    value0 = sum(c.amount for c in schedule.contributions() if c.date <= m["start"])
    res = simulate(cfg, schedule, market, m["start"], value0, market.rates[:, 0], record=(m["quote"]["date"],))
    return Backtest(starts, market, res)


def calibrate(cfg: dict, schedule: Schedule, bt: Backtest) -> Backtest:
    """Make each window's 2031-style quote from what was known at the time,
    all windows at once (windows x paths_per_window simulated futures)."""
    m, c = cfg["monte_carlo"], cfg["backtest"]["calibration"]
    qd, carve = m["quote"]["date"], cfg["decision_dates"]["reserve_carve_out"]
    k = month_starts(m["start"], carve).index(qd)
    value_then, _ = bt.result.recorded[qd]
    rate_then = bt.market.rates[:, k]
    ccfg = with_changes(cfg, {"monte_carlo.growth_returns.model": c["growth_model"],
                              "monte_carlo.rates.pull_per_year": 0.0})   # rates wander from where they are
    months = len(month_starts(qd, carve)) - 1
    w, p = len(bt.starts), c["paths_per_window"]
    draws = make_market(ccfg, months, w * p, m["seed"] + 2)
    res = simulate(ccfg, schedule, draws, qd, np.repeat(value_then, p), np.repeat(rate_then, p))
    fac = res.facility.reshape(w, p)
    bt.floor = np.percentile(fac, m["quote"]["floor_percentile"], axis=1)
    bt.ceiling = np.percentile(fac, m["quote"]["ceiling_percentile"], axis=1)
    return bt


def policy_menu(cfg: dict, schedule: Schedule, starts: list, market: Market) -> list:
    from .montecarlo import PolicyRun, policy_config
    runs = []
    for pol in cfg["monte_carlo"]["policies"]:
        for dial in cfg["monte_carlo"]["risk_dials"]:
            runs.append(PolicyRun(pol["code"], pol["name"], dial,
                                  replay(policy_config(cfg, pol, dial), schedule, starts, market).result))
    return runs
