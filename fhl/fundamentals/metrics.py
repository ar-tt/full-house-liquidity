"""Business and valuation measures, computed from raw filing numbers.

Plain-English glossary:
  FCF (free cash flow)   cash from operations minus spending on equipment:
                         money the business could hand to owners
  Invested capital       money tied up in the business (equity + debt - spare cash)
  ROIC                   after-tax operating profit / invested capital: the
                         return the business earns on the money it uses
  WACC                   the return investors demand for that money. ROIC above
                         WACC means the business creates value.
  EBITDA / EBIT          operating profit before (EBITDA) or after (EBIT)
                         depreciation; a rough, debt-neutral profit measure
  EV (enterprise value)  price of the whole business: shares + debt - cash
  Accrual ratio          how much of reported profit is NOT backed by cash;
                         high values are an early warning sign
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

from .annual import Financials, FiscalYear


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fcf(fy: FiscalYear):
    ocf, capex = fy.get("operating_cash_flow"), fy.get("capex")
    return None if ocf is None or capex is None else ocf - capex


def net_debt(fy: FiscalYear):
    debt, cash = fy.get("total_debt"), fy.get("cash")
    if debt is None or cash is None:
        return None
    return debt - cash - (fy.get("short_term_investments") or 0.0)


def ebit(fy: FiscalYear):
    """Operating profit. Some companies (e.g. J&J, Exxon) don't report an
    operating-income line; then pre-tax income + interest expense stands in."""
    op = fy.get("operating_income")
    if op is not None:
        return op
    pre, interest = fy.get("pretax_income"), fy.get("interest_expense")
    return None if pre is None else pre + (interest or 0.0)


def ebitda(fy: FiscalYear):
    op, da = ebit(fy), fy.get("depreciation_amortization")
    return None if op is None or da is None else op + da


def tax_rate(fy: FiscalYear, statutory: float) -> float:
    pre, tax = fy.get("pretax_income"), fy.get("income_tax")
    if pre is None or tax is None or pre <= 0:
        return statutory
    return min(max(tax / pre, 0.0), 0.5)


def invested_capital(fy: FiscalYear):
    eq, nd = fy.get("equity"), net_debt(fy)
    return None if eq is None or nd is None else eq + nd


def roic_series(fin: Financials, statutory: float, cap: float) -> list[tuple[date, float]]:
    """ROIC per year = operating profit after tax / average invested capital.
    If invested capital is zero or negative (common after big buybacks), the
    business needs no net capital, so ROIC counts as the cap."""
    out = []
    for prev, cur in zip(fin.years, fin.years[1:]):
        op = ebit(cur)
        ics = [invested_capital(prev), invested_capital(cur)]
        if op is None or None in ics:
            continue
        nopat = op * (1 - tax_rate(cur, statutory))
        ic = sum(ics) / 2
        roic = cap if ic <= 0 and nopat > 0 else (nopat / ic if ic > 0 else -cap)
        out.append((cur.end, min(roic, cap)))
    return out


def growth_rate(values: list) -> float | None:
    """Full-cycle growth per year, robust to freak years: the MEDIAN of the
    growth rates between every pair of years (Theil-Sen on the log values),
    so one odd year (a tax charge, a one-off gain) can't swing the answer.
    If any value is zero or negative, falls back to comparing 3-year
    averages at each end."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    vals = [v for _, v in pts]
    if len(vals) < 3:
        return None
    if all(v > 0 for v in vals):
        slopes = [(math.log(b) - math.log(a)) / (j - i)
                  for n, (i, a) in enumerate(pts) for j, b in pts[n + 1:]]
        return float(math.exp(np.median(slopes)) - 1)
    k = min(3, len(vals) // 2)
    start, end = _mean(vals[:k]), _mean(vals[-k:])
    span = len(vals) - k
    if start is None or end is None or start <= 0 or end <= 0 or span <= 0:
        return None
    return (end / start) ** (1 / span) - 1


def accrual_ratio(fin: Financials):
    """(net income - operating cash flow) / average total assets, latest year."""
    if len(fin.years) < 2:
        return None
    prev, cur = fin.years[-2], fin.years[-1]
    ni, ocf = cur.get("net_income"), cur.get("operating_cash_flow")
    assets = [prev.get("total_assets"), cur.get("total_assets")]
    if ni is None or ocf is None or None in assets or sum(assets) <= 0:
        return None
    return (ni - ocf) / (sum(assets) / 2)


def dividend_streak(dividends: list[tuple[date, float]], as_of: date) -> int:
    """Consecutive calendar years of higher total dividends per share,
    counted back from the last COMPLETE year before as_of."""
    by_year: dict[int, float] = {}
    for d, amt in dividends:
        if d <= as_of and d.year < as_of.year:
            by_year[d.year] = by_year.get(d.year, 0.0) + amt
    if not by_year:
        return 0
    years = sorted(by_year)
    streak, y = 0, years[-1]
    if y != as_of.year - 1:
        return 0  # stopped paying
    while y - 1 in by_year and by_year[y] > by_year[y - 1] * (1 + 1e-9):
        streak += 1
        y -= 1
    return streak


def growth_path(growth: float, years: int, terminal: float, fade_years: int = 0) -> list[float]:
    """Growth rate for each forecast year: `growth` for `years`, then stepping
    down in equal steps toward `terminal` over `fade_years` (the fade), after
    which the terminal rate applies forever."""
    fade = [growth + (terminal - growth) * i / (fade_years + 1) for i in range(1, fade_years + 1)]
    return [growth] * years + fade


def dcf_value(fcf0: float, growth: float, discount: float, years: int, terminal: float,
              fade_years: int = 0) -> float:
    """Value today of free cash flow that grows along growth_path(), then at
    `terminal` forever, discounted at `discount`."""
    if discount <= terminal:
        raise ValueError("discount rate must exceed terminal growth")
    pv, cash = 0.0, fcf0
    path = growth_path(growth, years, terminal, fade_years)
    for t, g in enumerate(path, start=1):
        cash *= 1 + g
        pv += cash / (1 + discount) ** t
    tv = cash * (1 + terminal) / (discount - terminal)
    return pv + tv / (1 + discount) ** len(path)


def implied_growth(ev: float, fcf0: float, discount: float, years: int, terminal: float,
                   lo: float, hi: float, fade_years: int = 0) -> float | None:
    """Reverse DCF: the growth rate the current price is betting on, using the
    same forecast shape (years + fade) as the forward DCF."""
    if fcf0 <= 0 or ev <= 0:
        return None
    f = lambda g: dcf_value(fcf0, g, discount, years, terminal, fade_years) - ev
    if f(lo) > 0:
        return lo      # even shrinking at `lo` is worth more than the price
    if f(hi) < 0:
        return hi      # the price needs more growth than we search for
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def position_in_range(current: float, history: list[float]) -> float | None:
    """Where `current` sits between the lowest (0) and highest (1) of history."""
    pts = [h for h in history if h is not None] + [current]
    if len(pts) < 3:
        return None
    lo, hi = min(pts), max(pts)
    return 0.5 if hi == lo else (current - lo) / (hi - lo)


def beta(stock: dict, market: dict, shrink: float, bounds: tuple, min_weeks: int = 104):
    """How much the stock moves when the market moves 1%, from weekly returns,
    pulled part-way toward 1 (the Blume adjustment) because extreme measured
    betas tend not to last."""
    weeks = sorted(set(stock) & set(market))
    if len(weeks) < min_weeks:
        return None
    s = np.array([stock[w] for w in weeks])
    m = np.array([market[w] for w in weeks])
    raw = float(np.cov(s, m)[0, 1] / np.var(m, ddof=1))
    adj = (1 - shrink) * raw + shrink * 1.0
    return min(max(adj, bounds[0]), bounds[1])
