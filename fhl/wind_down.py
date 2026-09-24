"""THE LONG HAND: how much of the portfolio may sit in growth assets, by date.

Three layers, applied in order:

1. CALENDAR. Each step of the schedule gives a band [low, high] for the
   growth share. HIGH is a hard ceiling on risk, never exceeded.

2. PROJECTED FUNDED STATUS decides where to sit:
     projected value at carve-out / projected cost of the reserve then.
   - behind (at or below `behind_below`): stay at the ceiling. Falling behind
     buys more time in equities, but never more than the ceiling and never
     borrowed money.
   - in between: slide from the top of the band to the bottom as it improves.
   - ahead (at or above `lock_in_above`): the goal is won, so lock the whole
     reserve in Treasuries now and keep only the surplus in growth.

3. HEDGE RAMP. From set dates, a rising share of the reserve's lock-in cost
   must already sit in Treasuries, so the carve-out never forces a sale of
   growth assets in a drawdown. It is waived when even an all-Treasury
   portfolio could not cover it, because locking in would lock in a shortfall.

4. RISK DIAL. The result is multiplied by `risk_dial` (0.75 = hold 75% of it),
   the setting step 5's Monte Carlo chose to meet the certainty test.

"Lock-in cost" = what the remaining payments would cost in Treasuries TODAY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .cashflows import Schedule
from .curves import Curve, year_fraction


def _row_on(rows: list[dict], d: date, key: str, default):
    current = default
    for r in sorted(rows, key=lambda r: r["from"]):
        if r["from"] <= d:
            current = r[key]
    return current


def band_on(cfg: dict, d: date) -> tuple[float, float]:
    rows = cfg["wind_down"]["schedule"]
    first = min(rows, key=lambda r: r["from"])["growth"]
    lo, hi = _row_on(rows, d, "growth", first)
    return float(lo), float(hi)


def hedge_share_on(cfg: dict, d: date) -> float:
    return float(_row_on(cfg["wind_down"]["hedge_ramp"], d, "share", 0.0))


def _payments(schedule: Schedule, cfg: dict, after: date):
    return [p for p in schedule.of_kind(*cfg["reserve"]["funds_kinds"]) if p.date >= after]


def lock_in_cost(schedule: Schedule, cfg: dict, as_of: date, curve: Curve) -> float:
    """What the remaining payments would cost in Treasuries bought today."""
    return sum(-p.amount * curve.discount_factor(year_fraction(as_of, p.date))
               for p in _payments(schedule, cfg, as_of))


def pv_future_contributions(schedule: Schedule, as_of: date, curve: Curve,
                            received_through: date | None = None) -> float:
    """Today's value of contributions not yet in the portfolio."""
    cutoff = max(as_of, received_through or as_of)
    return sum(c.amount * curve.discount_factor(year_fraction(as_of, c.date))
               for c in schedule.contributions() if c.date > cutoff)


def projected_reserve_cost(schedule: Schedule, cfg: dict, as_of: date, curve: Curve) -> float:
    """What the reserve is projected to cost on the carve-out date, from the
    rates the market sets today (the forward-implied cost)."""
    carve = cfg["decision_dates"]["reserve_carve_out"]
    t_carve = year_fraction(as_of, carve)
    df_carve = curve.discount_factor(t_carve)
    return sum(-p.amount * curve.discount_factor(year_fraction(as_of, p.date)) / df_carve
               for p in _payments(schedule, cfg, carve))


def projected_value(value: float, schedule: Schedule, cfg: dict, as_of: date,
                    received_through: date | None = None) -> float:
    """Value on the carve-out date if the portfolio follows the middle of the
    calendar band at the expected returns, plus promised contributions."""
    w = cfg["wind_down"]
    carve = cfg["decision_dates"]["reserve_carve_out"]
    marks = sorted({as_of, carve} | {date(y, 1, 1) for y in range(as_of.year + 1, carve.year + 1)})
    marks = [m for m in marks if as_of <= m <= carve]
    flows = {}
    cutoff = max(as_of, received_through or as_of)
    for c in schedule.contributions():
        if cutoff < c.date < carve:
            flows[c.date] = flows.get(c.date, 0.0) + c.amount
    v = value
    for a, b in zip(marks, marks[1:]):
        lo, hi = band_on(cfg, a)
        g = (lo + hi) / 2
        r = g * w["expected_return_growth"] + (1 - g) * w["expected_return_safe"]
        v = (v + flows.get(a, 0.0) if a != as_of else v) * (1 + r) ** year_fraction(a, b)
    return v


@dataclass
class WindDownStatus:
    as_of: date
    band: tuple
    value: float
    growth_value: float
    lock_cost: float = 0.0
    assets_with_contributions: float = 0.0
    projected_value: float = 0.0
    projected_reserve: float = 0.0
    funded_status: float | None = None
    hedge_share: float = 0.0
    target: float = 0.0
    reasons: list = field(default_factory=list)

    @property
    def growth_share(self) -> float:
        return self.growth_value / self.value if self.value else 0.0

    @property
    def ceiling(self) -> float:
        return self.band[1]

    @property
    def lock_ratio(self) -> float | None:
        """Assets (incl. promised contributions) / lock-in cost: 1 or more
        means the whole reserve could be locked in today."""
        return self.assets_with_contributions / self.lock_cost if self.lock_cost else None

    def action(self, tolerance: float) -> str:
        gap = self.growth_share - self.target
        if abs(gap) <= tolerance:
            return "on target"
        if gap > 0:
            return f"over target: move ${gap * self.value:,.0f} from growth assets into the Treasury ladder"
        return f"under target: room to add ${-gap * self.value:,.0f} of growth assets"


def status(value: float, growth_value: float, schedule: Schedule, cfg: dict, as_of: date,
           curve: Curve, received_through: date | None = None) -> WindDownStatus:
    """received_through: contributions dated on or before this are already in
    `value` (default: the as-of date)."""
    w = cfg["wind_down"]
    fs_cfg = w["funded_status"]
    carve = cfg["decision_dates"]["reserve_carve_out"]
    lo, hi = band_on(cfg, as_of)
    st = WindDownStatus(as_of, (lo, hi), value, growth_value)
    if as_of >= carve:
        st.target = 0.0
        st.reasons.append("the reserve has been carved out: no growth assets")
        return st

    st.lock_cost = lock_in_cost(schedule, cfg, as_of, curve)
    st.assets_with_contributions = value + pv_future_contributions(schedule, as_of, curve, received_through)
    st.projected_value = projected_value(value, schedule, cfg, as_of, received_through)
    st.projected_reserve = projected_reserve_cost(schedule, cfg, as_of, curve)
    fs = st.funded_status = st.projected_value / st.projected_reserve
    behind, ahead = fs_cfg["behind_below"], fs_cfg["lock_in_above"]

    # 2. funded status picks the spot in (or below) the band
    if fs <= behind:
        target = hi
        st.reasons.append(f"behind (projected funded status {fs:.2f} <= {behind:.2f}): "
                          f"stay at the {hi:.0%} ceiling, never above it")
    elif fs >= ahead and value > 0:
        target = min(lo, 1 - st.lock_cost / value)
        st.reasons.append(f"ahead (projected funded status {fs:.2f} >= {ahead:.2f}): lock the whole "
                          f"reserve (${st.lock_cost:,.0f}) in Treasuries, only the surplus stays in growth")
    else:
        target = hi - (fs - behind) / (ahead - behind) * (hi - lo)
        st.reasons.append(f"on track (projected funded status {fs:.2f}): "
                          f"{target:.0%} inside the {lo:.0%}-{hi:.0%} band")

    # 3. hedge ramp
    share = st.hedge_share = hedge_share_on(cfg, as_of)
    if share > 0 and value > 0:
        if st.lock_ratio is not None and st.lock_ratio < 1:
            target = hi
            st.reasons.append(f"hedge ramp ({share:.0%}) waived: assets cover only {st.lock_ratio:.0%} of "
                              "the lock-in cost, so locking in now would lock in a shortfall")
        else:
            cap = 1 - share * st.lock_cost / value
            if target > cap:
                target = cap
                st.reasons.append(f"hedge ramp: {share:.0%} of the reserve's lock-in cost "
                                  f"(${share * st.lock_cost:,.0f}) must already be in Treasuries")

    st.target = min(max(target, 0.0), hi)
    dial = w.get("risk_dial", 1.0)
    if dial != 1.0:
        st.reasons.append(f"risk dial {dial:.2f}: hold {dial:.0%} of the {st.target:.1%} the rules allow")
        st.target *= dial
    return st


def targets(schedule: Schedule, cfg: dict, as_of: date, value, rate, received_through: date | None = None):
    """status().target for many portfolios at once (the Monte Carlo's 10,000
    paths), each on a flat Treasury curve at its own `rate` (annual
    compounding). Same rules, same order; tests check it matches status()."""
    import numpy as np
    value = np.asarray(value, dtype=float)
    rate = np.asarray(rate, dtype=float)
    lo, hi = band_on(cfg, as_of)
    carve = cfg["decision_dates"]["reserve_carve_out"]
    if as_of >= carve:
        return np.zeros_like(value)
    fs_cfg = cfg["wind_down"]["funded_status"]
    behind, ahead = fs_cfg["behind_below"], fs_cfg["lock_in_above"]

    def pv(amounts, taus):
        if not len(amounts):
            return np.zeros_like(rate)
        return (np.asarray(amounts)[None, :] * (1 + rate[:, None]) ** (-np.asarray(taus)[None, :])).sum(axis=1)

    pays = _payments(schedule, cfg, as_of)
    faces = [-p.amount for p in pays]
    lock = pv(faces, [year_fraction(as_of, p.date) for p in pays])
    cutoff = max(as_of, received_through or as_of)
    future = [c for c in schedule.contributions() if c.date > cutoff]
    assets = value + pv([c.amount for c in future], [year_fraction(as_of, c.date) for c in future])
    b = projected_value(0.0, schedule, cfg, as_of, received_through)
    a = projected_value(1.0, schedule, cfg, as_of, received_through) - b
    carve_pays = _payments(schedule, cfg, carve)
    reserve_then = pv([-p.amount for p in carve_pays], [year_fraction(carve, p.date) for p in carve_pays])
    fs = (a * value + b) / reserve_then
    with np.errstate(divide="ignore", invalid="ignore"):
        lock_share = np.where(value > 0, lock / np.where(value > 0, value, 1), np.inf)
        slide = hi - (fs - behind) / (ahead - behind) * (hi - lo)
        target = np.where(fs <= behind, hi,
                          np.where((fs >= ahead) & (value > 0), np.minimum(lo, 1 - lock_share), slide))
        share = hedge_share_on(cfg, as_of)
        if share > 0:
            waived = assets / lock < 1
            target = np.where(value > 0, np.where(waived, hi, np.minimum(target, 1 - share * lock_share)), target)
    return np.clip(target, 0.0, hi) * cfg["wind_down"].get("risk_dial", 1.0)
