"""Historical data for backtests: monthly index returns and the Fed Treasury
curve on every date (which also gives the 3-month bill rate), loaded once."""
from __future__ import annotations

import bisect
import copy
import csv
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .curves import SvenssonCurve, _num, fetch_fed_curve_file
from .prices import Bar, YahooChartPrices


def month_key(d: date) -> tuple:
    return (d.year, d.month)


def monthly_returns_by_month(bars: list[Bar]) -> dict:
    """{(year, month): total return during that calendar month}."""
    level, month_end = 1.0, {}
    for prev, cur in zip(bars, bars[1:]):
        if prev.close > 0:
            level *= (cur.close + cur.dividend) / prev.close
        month_end[month_key(cur.date)] = level
    keys = sorted(month_end)
    return {k: month_end[k] / month_end[p] - 1 for p, k in zip(keys, keys[1:])}


def index_returns(cfg: dict, offline: bool = False) -> dict:
    b = cfg["backtest"]
    c = copy.deepcopy(cfg)
    c["prices"]["history_start"] = b["history_start"]
    c["prices"]["cache_subdir"] = b["cache_subdir"]
    return monthly_returns_by_month(YahooChartPrices(c, offline=offline).bars(b["series"], date.max))


class CurveHistory:
    """Every daily Fed zero curve, parsed once. params_on(d) gives the curve
    in force on d (the latest published on or before d)."""

    COLS = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")

    def __init__(self, cfg: dict, offline: bool = False):
        path = fetch_fed_curve_file(cfg) if not offline else \
            Path(cfg["_path"]).parent / cfg["data"]["cache_dir"] / cfg["data"]["treasury_curve"]["filename"]
        dates, params = [], []
        with open(path, newline="") as f:
            lines = iter(f)
            for line in lines:
                if line.startswith("Date,"):
                    header = next(csv.reader([line]))
                    break
            idx = [header.index(c) for c in self.COLS]
            for row in csv.reader(lines):
                if not row or not row[0]:
                    continue
                v = [_num(row[i]) for i in idx]
                if None in v[:3] or v[4] is None:
                    continue
                v[3] = v[3] or 0.0
                v[5] = v[5] if v[5] and v[5] > 0 else 1.0
                dates.append(datetime.strptime(row[0], "%Y-%m-%d").date())
                params.append(v)
        order = np.argsort(dates)
        self.dates = [dates[i] for i in order]
        self.params = np.array(params)[order]

    def params_on(self, days: list[date]) -> np.ndarray:
        idx = [bisect.bisect_right(self.dates, d) - 1 for d in days]
        if min(idx) < 0:
            raise ValueError(f"no Fed curve before {min(days)}")
        return self.params[idx]

    def curve_on(self, d: date) -> SvenssonCurve:
        p = self.params_on([d])[0]
        return SvenssonCurve(self.dates[bisect.bisect_right(self.dates, d) - 1], *p)


def zero_rates(params: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Continuously compounded zero rates (decimals) for many curves at once:
    params is (curves x 6), taus is (maturities,); result (curves x maturities)."""
    b0, b1, b2, b3, t1, t2 = (params[:, i][:, None] for i in range(6))
    t = np.maximum(taus[None, :], 1e-6)
    x1, x2 = t / t1, t / t2
    e1, e2 = np.exp(-x1), np.exp(-x2)
    y = b0 + b1 * (1 - e1) / x1 + b2 * ((1 - e1) / x1 - e1) + b3 * ((1 - e2) / x2 - e2)
    return y / 100


def flat_equivalent(params: np.ndarray, taus: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """For each curve, the single annual yield that prices these payments
    exactly as the curve does (so a flat-rate model reproduces the real price)."""
    target = (faces[None, :] * np.exp(-zero_rates(params, taus) * taus[None, :])).sum(axis=1)
    y = np.full(len(params), 0.05)
    for _ in range(50):
        disc = (1 + y[:, None]) ** (-taus[None, :])
        f = (faces[None, :] * disc).sum(axis=1) - target
        df = (faces[None, :] * -taus[None, :] * disc / (1 + y[:, None])).sum(axis=1)
        step = f / np.where(df == 0, -1e-12, df)
        y = np.maximum(y - step, -0.5)
        if np.max(np.abs(step)) < 1e-12:
            break
    return y
