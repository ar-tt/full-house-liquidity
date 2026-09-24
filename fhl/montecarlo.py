"""STEP 5: MONTE CARLO — the certainty test and the 2031 range.

We play the next six years forward 10,000 times, month by month. Each
"path" gets its own run of stock returns and its own drift in Treasury
yields, and the portfolio follows the engine's own rules along the way:

  - the wind-down target (calendar + funded status + hedge ramp),
  - scaled by the risk dial in config (1 = as designed, 0 = no growth assets),
  - capped by the risk controls: the volatility target, the 20% drawdown
    rule and the 50%-a-year turnover cap,
  - Laura's contributions arrive on their dates,
  - safe money buys the Treasury ladder first (bonds that pay exactly the
    ten payments), any extra sits in Treasury bills.

On 1 Jan 2033 the reserve is bought. A path SUCCEEDS if the portfolio
covers the whole reserve AND buying the part of the ladder not already
owned doesn't mean selling stocks while they are in a slump. Once the
reserve is fully funded, whatever is left is the facility contribution;
on a path that can't fund the reserve, the facility gets nothing.

Stock returns: real monthly VTI returns since 2001 (dividends included,
computed from daily prices), replayed in 12-month blocks so crashes stay
crashes, then shifted so the typical (median) year is the assumed +7%.
Treasury yields: one yield for all maturities, starting at today's Fed
10-year rate and drifting randomly toward a long-run level.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .cashflows import Schedule
from .config import ConfigError, with_changes
from .curves import year_fraction
from .prices import Bar, YahooChartPrices
from .wind_down import targets


# ------------------------------------------------------------------ inputs

def monthly_returns(bars: list[Bar]) -> np.ndarray:
    """Month-end to month-end total returns (dividends reinvested)."""
    level, month_end = 1.0, {}
    for prev, cur in zip(bars, bars[1:]):
        if prev.close > 0:
            level *= (cur.close + cur.dividend) / prev.close
        month_end[(cur.date.year, cur.date.month)] = level
    keys = sorted(month_end)
    vals = np.array([month_end[k] for k in keys])
    return vals[1:] / vals[:-1] - 1


def load_history(cfg: dict, offline: bool = False) -> np.ndarray:
    g = cfg["monte_carlo"]["growth_returns"]
    c = copy.deepcopy(cfg)
    c["prices"]["history_start"] = g["history_start"]
    c["prices"]["cache_subdir"] = g["cache_subdir"]
    bars = YahooChartPrices(c, offline=offline).bars(g["series"], date.max)
    return monthly_returns(bars)


@dataclass
class Market:
    """The futures every policy is run on. Random draws for the Monte Carlo;
    for a historical backtest each "path" is one real window of history, and
    the optional fields carry what actually happened."""
    growth: np.ndarray        # paths x months, simple returns of growth assets
    rate_shocks: np.ndarray   # paths x months, standard normal draws
    long_run_vol: float       # growth-asset volatility per year
    rates: np.ndarray | None = None   # paths x (months+1): actual yields, used instead of random drift
    bills: np.ndarray | None = None   # paths x months: actual monthly T-bill returns for cash
    prior: np.ndarray | None = None   # paths x lookback: growth returns BEFORE the start (for the vol target)

    @property
    def paths(self) -> int:
        return self.growth.shape[0]


def make_market(cfg: dict, months: int, paths: int, seed: int, history: np.ndarray | None = None) -> Market:
    g = cfg["monte_carlo"]["growth_returns"]
    rng = np.random.default_rng(seed)
    centre = math.log(1 + g["median_annual_return"]) / 12
    if g["model"] == "bootstrap":
        if history is None or len(history) < 2 * g["block_months"]:
            raise ConfigError("bootstrap needs a return history of at least two blocks")
        logs = np.log1p(history)
        logs = logs - logs.mean() + centre
        block = g["block_months"]
        n_blocks = -(-months // block)
        starts = rng.integers(0, len(logs) - block + 1, size=(paths, n_blocks))
        idx = (starts[..., None] + np.arange(block)).reshape(paths, n_blocks * block)[:, :months]
        growth = np.expm1(logs[idx])
        vol = float(logs.std(ddof=1) * math.sqrt(12))
    elif g["model"] == "lognormal":
        vol = g["lognormal_vol"]
        growth = np.expm1(rng.normal(centre, vol / math.sqrt(12), size=(paths, months)))
    else:
        raise ConfigError(f"monte_carlo.growth_returns.model must be bootstrap or lognormal, got '{g['model']}'")
    return Market(growth, rng.standard_normal((paths, months)), vol)


def month_starts(start: date, end: date) -> list[date]:
    if start.day != 1:
        raise ConfigError(f"the simulation starts on the 1st of a month, got {start}")
    out, d = [], start
    while d <= end:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def vol_target_on(cfg: dict, d: date) -> float:
    rows = sorted(cfg["risk_controls"]["vol_target"], key=lambda r: r["from"])
    current = rows[0]["vol"]
    for r in rows:
        if r["from"] <= d:
            current = r["vol"]
    return current


# ------------------------------------------------------------------ simulation

@dataclass
class SimResult:
    dates: list
    total: np.ndarray         # portfolio value on the carve-out date
    reserve: np.ndarray       # what the reserve cost that day
    funded: np.ndarray        # portfolio covered the whole reserve
    forced: np.ndarray        # had to sell stocks in a slump to finish the ladder
    facility: np.ndarray      # left for the facility (0 if the reserve wasn't covered)
    ladder_owned: np.ndarray  # share of the ladder already owned before the carve-out
    growth_share: np.ndarray  # months x paths, share in growth assets after each month's trades
    vol_capped: np.ndarray    # months: share of paths where the volatility target was binding
    dd_capped: np.ndarray     # months: share of paths under the drawdown rule
    turnover: np.ndarray      # months x paths, share of the portfolio traded (new money excluded)
    recorded: dict = field(default_factory=dict)   # date -> (values, rates)

    @property
    def success(self) -> np.ndarray:
        return self.funded & ~self.forced

    def p(self, arr) -> float:
        return float(np.mean(arr))

    def pct(self, q) -> float:
        return float(np.percentile(self.facility, q))


def _lock(schedule, cfg, d, rate) -> np.ndarray:
    pays = [p for p in schedule.of_kind(*cfg["reserve"]["funds_kinds"]) if p.date >= d]
    faces = np.array([-p.amount for p in pays])
    taus = np.array([year_fraction(d, p.date) for p in pays])
    return (faces[None, :] * (1 + rate[:, None]) ** (-taus[None, :])).sum(axis=1)


def simulate(cfg: dict, schedule: Schedule, market: Market, start: date, value0: float,
             rate0: float, record: tuple = ()) -> SimResult:
    """Run every path from `start` (portfolio worth `value0`, all in cash,
    contributions dated on or before `start` already included) to the carve-out."""
    rc, mc = cfg["risk_controls"], cfg["monte_carlo"]
    rates = mc["rates"]
    carve = cfg["decision_dates"]["reserve_carve_out"]
    dates = month_starts(start, carve)
    months = len(dates) - 1
    if market.growth.shape[1] < months:
        raise ConfigError(f"market has {market.growth.shape[1]} months, the run needs {months}")
    n = market.paths
    contrib = {}
    for c in schedule.contributions():
        if start < c.date < carve:
            if c.date not in dates:
                raise ConfigError(f"contribution on {c.date} is not on the 1st of a month")
            contrib[c.date] = contrib.get(c.date, 0.0) + c.amount

    G, owned = np.zeros(n), np.zeros(n)
    C = np.broadcast_to(np.asarray(value0, dtype=float), (n,)).copy()   # one value, or one per path
    y = np.broadcast_to(np.asarray(rate0, dtype=float), (n,)).copy()
    idx, peak, dd_mode = np.ones(n), np.ones(n), np.zeros(n, dtype=bool)
    g_idx, g_peak = np.ones(n), np.ones(n)
    look = rc["vol_lookback_months"]
    recent = np.zeros((n, look))
    if market.prior is not None:
        recent = np.log1p(market.prior[:, -look:]).copy()
    if market.rates is not None:
        y = market.rates[:, 0].copy()
    turnover = np.zeros((n, 12))
    shares, vol_capped, dd_capped, traded, recorded = [], [], [], [], {}

    for k in range(months):
        d = dates[k]
        new_money = contrib.get(d, 0.0)
        C += new_money
        lock = _lock(schedule, cfg, d, y)
        V = G + C + owned * lock
        if d in record:
            recorded[d] = (V.copy(), y.copy())

        # target growth share: the live wind-down rule (risk dial included), then the risk controls
        tgt = targets(schedule, cfg, d, V, y, received_through=d)
        known = k >= look or market.prior is not None
        sigma = recent.std(axis=1, ddof=1) * math.sqrt(12) if known else np.full(n, market.long_run_vol)
        vol_cap = vol_target_on(cfg, d) / np.maximum(sigma, 1e-9)
        vol_capped.append(float(np.mean(vol_cap < tgt)))
        tgt = np.minimum(tgt, vol_cap)
        dd_capped.append(float(np.mean(dd_mode)))
        tgt = np.where(dd_mode, np.minimum(tgt, rc["drawdown_growth_cap"]), tgt)
        tgt = np.clip(tgt, 0.0, 1.0)

        # trade toward it, within the turnover cap (new money isn't turnover)
        delta = tgt * V - G
        exempt = V if k == 0 else np.full(n, new_money)
        size = np.abs(delta)
        counted = np.maximum(size - exempt, 0.0)
        budget = np.maximum(rc["turnover_cap"] - turnover[:, 1:].sum(axis=1), 0.0) * V
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(counted > budget, (np.minimum(size, exempt) + budget) / size, 1.0)
        delta = delta * np.nan_to_num(scale, nan=1.0)
        turnover = np.roll(turnover, -1, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            turnover[:, -1] = np.where(V > 0, np.maximum(np.abs(delta) - exempt, 0.0) / V, 0.0)
        traded.append(turnover[:, -1].copy())
        G = G + delta
        safe = V - G
        owned = np.minimum(1.0, safe / lock)            # ladder first ...
        C = safe - owned * lock                         # ... the rest in Treasury bills
        shares.append(np.where(V > 0, G / V, 0.0))

        # one month passes
        r = market.growth[:, k]
        G = G * (1 + r)
        C = C * (1 + (market.bills[:, k] if market.bills is not None else (1 + y) ** (1 / 12) - 1))
        if market.rates is not None:
            y = market.rates[:, k + 1]
        else:
            y = np.maximum(rates["floor"], y + rates["pull_per_year"] * (rates["long_run"] - y) / 12
                           + rates["vol_per_year"] / math.sqrt(12) * market.rate_shocks[:, k])
        V_after = G + C + owned * _lock(schedule, cfg, dates[k + 1], y)
        idx = idx * V_after / V
        peak = np.maximum(peak, idx)
        dd_mode = np.where(dd_mode, idx < peak, idx <= peak * (1 - rc["drawdown_trigger"]))
        g_idx = g_idx * (1 + r)
        g_peak = np.maximum(g_peak, g_idx)
        recent = np.roll(recent, -1, axis=1)
        recent[:, -1] = np.log1p(r)

    if carve in record:
        recorded[carve] = (None, y.copy())
    reserve = _lock(schedule, cfg, carve, y)
    total = G + C + owned * reserve
    funded = total >= reserve - 1e-6
    need_from_stocks = np.maximum((1 - owned) * reserve - C, 0.0)
    slump = 1 - g_idx / g_peak >= mc["forced_sale_drawdown"]
    forced = funded & (need_from_stocks > 1e-6) & slump
    facility = np.where(funded, total - reserve, 0.0)
    return SimResult(dates, total, reserve, funded, forced, facility, owned, np.array(shares),
                     np.array(vol_capped), np.array(dd_capped), np.array(traded), recorded)


# ------------------------------------------------------------------ the three outputs

def start_rate(cfg: dict, curve) -> float:
    r = cfg["monte_carlo"]["rates"]
    if r["start"] != "from_curve":
        return float(r["start"])
    if curve is None:
        return float(r["long_run"])
    return math.exp(curve.zero_rate(r["start_maturity_years"])) - 1


@dataclass
class PolicyRun:
    code: str
    name: str
    dial: float
    result: SimResult

    @property
    def success(self) -> float:
        return self.result.p(self.result.success)

    @property
    def median(self) -> float:
        return self.result.pct(50)


def policy_menu(cfg: dict, schedule: Schedule, market: Market, start: date, value0: float,
                rate0: float) -> list[PolicyRun]:
    runs = []
    for pol in cfg["monte_carlo"]["policies"]:
        for dial in cfg["monte_carlo"]["risk_dials"]:
            pcfg = policy_config(cfg, pol, dial)
            runs.append(PolicyRun(pol["code"], pol["name"], dial,
                                  simulate(pcfg, schedule, market, start, value0, rate0)))
    return runs


def policy_config(cfg: dict, policy: dict, dial: float) -> dict:
    return with_changes(cfg, {**policy["changes"], "wind_down.risk_dial": dial})


def best_for(runs: list[PolicyRun], threshold: float) -> PolicyRun | None:
    """The policy that leaves Laura the most facility money (median) while
    still meeting the certainty threshold."""
    ok = [r for r in runs if r.success >= threshold - 1e-12]
    return max(ok, key=lambda r: (r.median, r.success)) if ok else None


@dataclass
class Quote:
    as_of: date
    start_value: float
    floor: float
    ceiling: float
    floor_pct: int
    ceiling_pct: int
    result: SimResult

    def sentence(self) -> str:
        return (f"{100 - self.floor_pct}% confident the contribution will be at least ${self.floor:,.0f}; "
                f"{self.ceiling_pct - self.floor_pct}% chance it lands between ${self.floor:,.0f} "
                f"and ${self.ceiling:,.0f}")


def quote(cfg: dict, schedule: Schedule, market: Market, as_of: date, value: float, rate: float) -> Quote:
    """The co-sponsor range as it would be quoted on `as_of` from a portfolio
    worth `value`: the floor and ceiling percentiles of what is left for the
    facility AFTER the reserve is fully funded."""
    q = cfg["monte_carlo"]["quote"]
    res = simulate(cfg, schedule, market, as_of, value, rate)
    return Quote(as_of, value, res.pct(q["floor_percentile"]), res.pct(q["ceiling_percentile"]),
                 q["floor_percentile"], q["ceiling_percentile"], res)
