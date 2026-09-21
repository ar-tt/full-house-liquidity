"""Interest-rate curves: turn "a dollar paid in t years" into "what it costs today".

Every curve answers one question, discount_factor(t): the price today of $1
paid t years from now. A $50,000 payment in 3 years costs 50,000 * DF(3).

  FlatCurve      one rate for every maturity (how the anchor numbers work)
  SvenssonCurve  the Federal Reserve's fitted Treasury zero-coupon curve
  ForwardCurve   a curve "seen from" a future date, e.g. what today's market
                 implies 2033 prices will be
"""
from __future__ import annotations

import csv
import math
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


def year_fraction(start: date, end: date) -> float:
    """Years between two dates. Jan 1 to Jan 1 is always a whole number."""
    def frac(d: date) -> float:
        days_in_year = (date(d.year + 1, 1, 1) - date(d.year, 1, 1)).days
        return d.year + (d - date(d.year, 1, 1)).days / days_in_year
    return frac(end) - frac(start)


class Curve:
    name = "curve"

    def discount_factor(self, t: float) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


@dataclass
class FlatCurve(Curve):
    rate: float
    compounding: str = "annual"  # annual | semiannual | continuous

    def __post_init__(self):
        if self.compounding not in ("annual", "semiannual", "continuous"):
            raise ValueError(f"unknown compounding '{self.compounding}'")
        self.name = f"flat {self.rate:.2%} ({self.compounding})"

    def discount_factor(self, t: float) -> float:
        if self.compounding == "annual":
            return (1 + self.rate) ** -t
        if self.compounding == "semiannual":
            return (1 + self.rate / 2) ** (-2 * t)
        return math.exp(-self.rate * t)


@dataclass
class SvenssonCurve(Curve):
    """Gürkaynak-Sack-Wright (Federal Reserve) fitted zero curve for one date.

    The Fed publishes six parameters per day; from them we get the zero-coupon
    yield for ANY maturity, continuously compounded, in percent.
    """
    as_of: date
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def __post_init__(self):
        self.name = f"Fed Treasury zero curve as of {self.as_of}"

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero yield as a decimal (0.047 = 4.7%)."""
        if t <= 0:
            t = 1e-6  # limit as t -> 0 is beta0 + beta1
        x1 = t / self.tau1
        e1 = math.exp(-x1)
        y = self.beta0 + self.beta1 * (1 - e1) / x1 + self.beta2 * ((1 - e1) / x1 - e1)
        if self.beta3 and self.tau2 and self.tau2 > 0:
            x2 = t / self.tau2
            e2 = math.exp(-x2)
            y += self.beta3 * ((1 - e2) / x2 - e2)
        return y / 100.0

    def discount_factor(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return math.exp(-self.zero_rate(t) * t)


@dataclass
class ForwardCurve(Curve):
    """A curve viewed from `offset` years in the future.

    DF_forward(t) = DF(offset + t) / DF(offset). This is the price that can be
    locked in today for $1 paid t years after the future date.
    """
    base: Curve
    offset: float

    def __post_init__(self):
        self.name = f"forward-implied from [{self.base.describe()}], {self.offset:.2f}y ahead"

    def discount_factor(self, t: float) -> float:
        return self.base.discount_factor(self.offset + t) / self.base.discount_factor(self.offset)


# ---------------------------------------------------------------------------
# Loading the Fed curve (free, daily, point-in-time: you can ask for the curve
# as it stood on any past date, which the backtests will need)
# ---------------------------------------------------------------------------

def fetch_fed_curve_file(cfg: dict, force: bool = False) -> Path:
    data = cfg["data"]
    src = data["treasury_curve"]
    cache_dir = Path(cfg["_path"]).parent / data["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / src["filename"]
    max_age = data["cache_max_age_days"] * 86400
    if force or not path.exists() or time.time() - path.stat().st_mtime > max_age:
        req = urllib.request.Request(src["url"], headers={"User-Agent": "fhl-quant-engine"})
        with urllib.request.urlopen(req, timeout=data["request_timeout_seconds"]) as resp:
            body = resp.read()
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
    return path


def _num(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def load_fed_curve(path: str | Path, as_of: date | None = None) -> SvenssonCurve:
    """The Fed curve for the latest published date on or before `as_of`
    (latest overall if as_of is None)."""
    best = None
    with open(path, newline="") as f:
        lines = iter(f)
        for line in lines:  # skip the notes above the real header
            if line.startswith("Date,"):
                header = next(csv.reader([line]))
                break
        else:
            raise ValueError(f"{path} does not look like the Fed curve file (no 'Date,' header)")
        idx = {name: i for i, name in enumerate(header)}
        for row in csv.reader(lines):
            if not row or not row[0]:
                continue
            d = datetime.strptime(row[0], "%Y-%m-%d").date()
            if as_of and d > as_of:
                continue
            vals = {k: _num(row[idx[k]]) for k in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")}
            if None in (vals["BETA0"], vals["BETA1"], vals["BETA2"], vals["TAU1"]):
                continue
            if best is None or d > best[0]:
                best = (d, vals)
    if best is None:
        raise ValueError(f"no usable curve in {path} on or before {as_of}")
    d, v = best
    return SvenssonCurve(d, v["BETA0"], v["BETA1"], v["BETA2"], v["BETA3"] or 0.0,
                         v["TAU1"], v["TAU2"] or 0.0)
