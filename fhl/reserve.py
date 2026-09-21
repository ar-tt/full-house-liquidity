"""Operating reserve: price the Treasury ladder and show how it runs down.

The reserve is one zero-coupon Treasury ("rung") per payment. Each rung pays
exactly one $50,000 payment on its date, so after the carve-out the payments
no longer depend on markets at all: every bond is held until it pays out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .cashflows import CashFlow, Schedule
from .config import ConfigError
from .curves import Curve, FlatCurve, ForwardCurve, SvenssonCurve, year_fraction

PRICING_METHODS = ("flat", "curve_spot", "curve_forward")


@dataclass
class Rung:
    payment: CashFlow
    years_to_maturity: float
    discount_factor: float

    @property
    def face(self) -> float:
        return -self.payment.amount

    @property
    def cost(self) -> float:
        return self.face * self.discount_factor

    @property
    def locked_yield(self) -> float:
        """Annually compounded yield this rung earns if held to maturity."""
        if self.years_to_maturity <= 0:
            return 0.0
        return self.discount_factor ** (-1 / self.years_to_maturity) - 1

    def book_value(self, years_elapsed: float) -> float:
        """Value after `years_elapsed`, growing steadily at its locked yield.
        Market value will wobble with rates, but a bond held to maturity always
        ends at face, so book value is what matters for funding payments."""
        remaining = self.years_to_maturity - years_elapsed
        if remaining <= 0:
            return self.face
        return self.face * self.discount_factor ** (remaining / self.years_to_maturity)


@dataclass
class RunoffRow:
    date: date
    value_before: float
    payment: float
    value_after: float
    rungs_left: int
    face_left: float
    avg_years_left: float           # value-weighted years to maturity after payment
    interest_next_year: float       # how much the remaining bonds grow by next payment
    composition: dict = field(default_factory=dict)  # bucket label -> share of value after


@dataclass
class Ladder:
    carve_date: date
    curve: Curve
    rungs: list[Rung]

    @property
    def cost(self) -> float:
        return sum(r.cost for r in self.rungs)

    @property
    def total_face(self) -> float:
        return sum(r.face for r in self.rungs)

    def composition(self, buckets: list[dict], at_years: float = 0.0,
                    include_paid: bool = True) -> dict:
        """Share of reserve value in each maturity bucket, `at_years` after carve-out."""
        values = {b["label"]: 0.0 for b in buckets}
        for r in self.rungs:
            left = r.years_to_maturity - at_years
            if left < -1e-9 or (abs(left) <= 1e-9 and not include_paid):
                continue
            values[_bucket(buckets, max(left, 0.0))] += r.book_value(at_years)
        total = sum(values.values())
        return {k: (v / total if total else 0.0) for k, v in values.items()}

    def runoff(self, buckets: list[dict]) -> list[RunoffRow]:
        rows = []
        pay_dates = sorted({r.payment.date for r in self.rungs})
        for i, d in enumerate(pay_dates):
            s = year_fraction(self.carve_date, d)
            live = [r for r in self.rungs if r.payment.date >= d]
            before = sum(r.book_value(s) for r in live)
            paid = sum(r.face for r in live if r.payment.date == d)
            after_rungs = [r for r in live if r.payment.date > d]
            after = sum(r.book_value(s) for r in after_rungs)
            if after_rungs:
                avg = sum((r.years_to_maturity - s) * r.book_value(s) for r in after_rungs) / after
                nxt = year_fraction(self.carve_date, pay_dates[i + 1])
                interest = sum(r.book_value(nxt) for r in after_rungs) - after
            else:
                avg, interest = 0.0, 0.0
            rows.append(RunoffRow(
                date=d, value_before=before, payment=paid,
                value_after=after, rungs_left=len(after_rungs),
                face_left=sum(r.face for r in after_rungs), avg_years_left=avg,
                interest_next_year=interest,
                composition=self.composition(buckets, at_years=s, include_paid=False) if after_rungs else {},
            ))
        return rows


def _bucket(buckets: list[dict], years_left: float) -> str:
    for b in buckets:
        if years_left <= b["max_years"] + 1e-9:
            return b["label"]
    return buckets[-1]["label"]


def build_ladder(schedule: Schedule, reserve_cfg: dict, carve_date: date, curve: Curve) -> Ladder:
    funded = schedule.of_kind(*reserve_cfg["funds_kinds"])
    if not funded:
        raise ConfigError(f"no cash_flows rows of kind {reserve_cfg['funds_kinds']} for the reserve to fund")
    early = [r for r in funded if r.date < carve_date]
    if early:
        raise ConfigError(f"payment(s) dated before the reserve carve-out {carve_date}: "
                          f"{[str(r.date) for r in early]}")
    if any(r.is_inflow for r in funded):
        raise ConfigError("reserve.funds_kinds includes a contribution row; it should only list payments")
    rungs = []
    for p in funded:
        t = year_fraction(carve_date, p.date)
        rungs.append(Rung(p, t, curve.discount_factor(t)))
    return Ladder(carve_date, curve, rungs)


def reserve_curve(cfg: dict, method: str | None = None, rate: float | None = None,
                  fed_curve: SvenssonCurve | None = None) -> Curve:
    """The curve used to price the reserve on the carve-out date."""
    r = cfg["reserve"]
    method = method or r["pricing"]
    if method not in PRICING_METHODS:
        raise ConfigError(f"reserve.pricing must be one of {PRICING_METHODS}, got '{method}'")
    if method == "flat":
        return FlatCurve(r["flat_rate"] if rate is None else rate, r["compounding"])
    if fed_curve is None:
        raise ConfigError(f"reserve.pricing '{method}' needs the Fed Treasury curve, which is not loaded")
    if method == "curve_spot":
        return fed_curve
    carve = cfg["decision_dates"]["reserve_carve_out"]
    return ForwardCurve(fed_curve, year_fraction(fed_curve.as_of, carve))
