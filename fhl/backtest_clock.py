"""BACKTEST 2: does the Position Clock's timing beat buying all at once?

For every trading day, pretend the engine decided that day to buy $1 of an
index ETF. Three ways in:
  LUMP    all $1 at that day's close
  THIRDS  a third now, a third 3 weeks later, a third 6 weeks later
  CLOCK   the live Position Clock rules: tranche spacing from the trend regime
          (200-day average + ADX), RSI < 40 pulls a tranche a week earlier,
          RSI > 70 holds it back, everything in by 8 weeks after the decision
Each day's decision uses only indicators from the day BEFORE (no peeking),
and money waiting to be invested earns the 3-month Treasury rate. Then the
three are compared 3 and 12 months after the decision.

The 8-week window is counted from the decision day. (In the live engine,
record the decision in portfolio.yaml `entries` with tranches_done: 0 to get
the same anchor; otherwise a first tranche held back by RSI > 70 has no
deadline.)
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .history import CurveHistory, zero_rates
from .indicators import adx_series, rsi_series, sma_series
from .prices import Bar, YahooChartPrices

REGIMES = ("uptrend", "no_trend", "downtrend")


def regimes(bars: list[Bar], icfg: dict) -> np.ndarray:
    closes = np.array([b.close for b in bars])
    ma = sma_series(closes, icfg["trend_ma_days"])
    ax = adx_series(bars, icfg["adx_days"])
    out = np.full(len(bars), "no_trend", dtype=object)
    trending = ~np.isnan(ma) & ~np.isnan(ax) & (np.nan_to_num(ax) > icfg["adx_trending_above"])
    out[trending & (closes > np.nan_to_num(ma))] = "uptrend"
    out[trending & (closes <= np.nan_to_num(ma))] = "downtrend"
    return out


def total_return_index(bars: list[Bar]) -> np.ndarray:
    tr = np.ones(len(bars))
    for i in range(1, len(bars)):
        tr[i] = tr[i - 1] * (bars[i].close + bars[i].dividend) / bars[i - 1].close
    return tr


@dataclass
class Entries:
    ticker: str
    dates: list               # decision days
    regime: np.ndarray        # regime the day before the decision
    rsi: np.ndarray           # RSI the day before the decision
    lump: dict                # horizon -> value of $1 at the horizon
    thirds: dict
    clock: dict
    clock_first_delay: np.ndarray   # trading days before the clock's first tranche
    clock_price: np.ndarray   # clock's average purchase price / decision-day price
    thirds_price: np.ndarray


def run(bars: list[Bar], bills: np.ndarray, cfg: dict, ticker: str = "") -> Entries:
    """bills[i] = interest earned by cash held from day i to day i+1."""
    pc, b = cfg["position_clock"], cfg["backtest"]["clock"]
    icfg = pc["indicators"]
    horizons = b["horizons_days"]
    n_tr = pc["tranches"]
    closes = np.array([x.close for x in bars])
    days = [x.date for x in bars]
    tr = total_return_index(bars)
    reg = regimes(bars, icfg)
    rs = rsi_series(closes, icfg["rsi_days"])
    spacing = {k: timedelta(weeks=v["spacing_weeks"]) for k, v in pc["regimes"].items()}
    accel, window = timedelta(weeks=pc["accelerate_weeks"]), timedelta(weeks=pc["max_window_weeks"])
    naive = timedelta(weeks=b["naive_spacing_weeks"])
    first = icfg["trend_ma_days"] + 1
    last = len(bars) - max(horizons) - 1
    out = {k: {h: [] for h in horizons} for k in ("lump", "thirds", "clock")}
    starts, delays, cprice, tprice = [], [], [], []

    def value_at(buys, cash_left, h_idx):
        return sum(units * tr[h_idx] for units in buys) + cash_left

    for i in range(first, last):
        d0 = days[i]
        # THIRDS: fixed dates
        cash, units, spent = 1.0, [], []
        targets = [d0 + k * naive for k in range(n_tr)]
        k = 0
        j = i
        while k < n_tr:
            if days[j] >= targets[k]:
                amt = cash if k == n_tr - 1 else 1.0 / n_tr
                units.append(amt / tr[j])
                spent.append((amt, closes[j]))
                cash -= amt
                k += 1
                continue
            cash *= 1 + bills[j]
            j += 1
        thirds_units, thirds_spent = units, spent
        # CLOCK: the live rules, evaluated each day with yesterday's signals
        cash, units, spent, k, j = 1.0, [], [], 1, i
        first_buy = None
        while k <= n_tr:
            r, g = rs[j - 1], reg[j - 1]
            due = d0 + (k - 1) * spacing[g] - (accel if (not math.isnan(r) and r < icfg["rsi_accelerate_below"]) else timedelta(0))
            late = days[j] >= d0 + window
            hot = not math.isnan(r) and r > icfg["rsi_delay_above"]
            if late or (not hot and days[j] >= due):
                amt = cash if k == n_tr or late else 1.0 / n_tr
                units.append(amt / tr[j])
                spent.append((amt, closes[j]))
                cash -= amt
                first_buy = j if first_buy is None else first_buy
                k = n_tr + 1 if late else k + 1
                continue
            cash *= 1 + bills[j]
            j += 1
        starts.append(d0)
        delays.append(first_buy - i)
        cprice.append(sum(a for a, _ in spent) / sum(a / p for a, p in spent) / closes[i])
        tprice.append(sum(a for a, _ in thirds_spent) / sum(a / p for a, p in thirds_spent) / closes[i])
        for h in horizons:
            e = i + h
            out["lump"][h].append(tr[e] / tr[i])
            out["thirds"][h].append(value_at(thirds_units, 0.0, e))
            out["clock"][h].append(value_at(units, 0.0, e))
    idx = np.arange(first, last)
    return Entries(ticker, starts, reg[idx - 1], rs[idx - 1],
                   {h: np.array(v) for h, v in out["lump"].items()},
                   {h: np.array(v) for h, v in out["thirds"].items()},
                   {h: np.array(v) for h, v in out["clock"].items()},
                   np.array(delays), np.array(cprice), np.array(tprice))


def load(cfg: dict, ticker: str, curves: CurveHistory, offline: bool = False) -> tuple[list, np.ndarray]:
    c = copy.deepcopy(cfg)
    c["prices"]["history_start"] = cfg["backtest"]["history_start"]
    c["prices"]["cache_subdir"] = cfg["backtest"]["cache_subdir"]
    bars = YahooChartPrices(c, offline=offline).bars(ticker, date.max)
    bars = [b for b in bars if b.date >= curves.dates[0]]
    z = zero_rates(curves.params_on([b.date for b in bars]), np.array([cfg["backtest"]["bill_maturity_years"]]))[:, 0]
    return bars, np.expm1(z / 252)
