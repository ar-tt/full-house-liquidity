"""The historical backtest, on small made-up histories we control."""
import math
from datetime import date

import numpy as np
import pytest

from fhl import backtest as bk
from fhl.config import ConfigError, with_changes
from fhl.curves import load_fed_curve
from fhl.history import CurveHistory, flat_equivalent, monthly_returns_by_month, zero_rates
from fhl.montecarlo import month_starts
from fhl.prices import Bar
from conftest import FIXTURES

RESERVE_AT_4 = 421_766.58


class FlatCurves:
    """Stand-in for CurveHistory: the same flat curve on every date."""

    def __init__(self, cont_rate, first=date(1970, 1, 1), last=date(2030, 1, 1)):
        self.dates = [first, last]
        self.p = [cont_rate * 100, 0.0, 0.0, 0.0, 1.0, 1.0]

    def params_on(self, days):
        return np.array([self.p] * len(days))


def returns_dict(first, last, value=0.0):
    out, d = {}, first
    while d <= last:
        out[(d.year, d.month)] = value
        d = bk.add_months(d, 1)
    return out


def test_add_months():
    assert bk.add_months(date(2026, 11, 1), 3) == date(2027, 2, 1)
    assert bk.add_months(date(2027, 1, 1), -1) == date(2026, 12, 1)


def test_monthly_returns_by_month_adds_dividends():
    bars = [Bar(date(2019, 12, 31), 100, 100, 100, 100, 1), Bar(date(2020, 1, 31), 100, 100, 100, 100, 1),
            Bar(date(2020, 2, 28), 99, 99, 99, 99, 1, dividend=2.0), Bar(date(2020, 3, 31), 110, 110, 110, 110, 1)]
    got = monthly_returns_by_month(bars)
    assert got[(2020, 2)] == pytest.approx(0.01)            # price -1, dividend +2 on a 100 start
    assert got[(2020, 3)] == pytest.approx(110 / 99 - 1)
    assert (2019, 12) not in got                            # the first month has nothing before it


def test_zero_rates_match_the_single_curve_formula():
    ch = CurveHistory.__new__(CurveHistory)
    c = load_fed_curve(FIXTURES / "gsw_sample.csv")
    params = np.array([[c.beta0, c.beta1, c.beta2, c.beta3, c.tau1, c.tau2]])
    taus = np.array([0.25, 1, 5, 10, 16])
    assert zero_rates(params, taus)[0] == pytest.approx([c.zero_rate(t) for t in taus], abs=1e-12)


def test_flat_equivalent_of_a_flat_curve_is_its_annual_rate():
    p = np.array([[4.0, 0, 0, 0, 1, 1]])                               # 4% continuous, all maturities
    y = flat_equivalent(p, np.arange(6.0, 16.0), np.full(10, 50_000.0))
    assert y[0] == pytest.approx(math.exp(0.04) - 1, abs=1e-12)


def test_windows_skip_gaps_and_stop_where_the_data_stops(cfg):
    months = 72
    rets = returns_dict(date(1979, 1, 1), date(1995, 12, 1))
    del rets[(1982, 5)]                                                # a hole in the data
    c = with_changes(cfg, {"backtest.first_start": date(1980, 1, 1)})
    starts = bk.window_starts(c, rets, FlatCurves(0.04, last=date(1992, 1, 1)), months)
    # a window needs its 12 months before day one plus its 72 months, so the
    # hole in May 1982 rules out every day one up to May 1983
    assert starts[0] == date(1983, 6, 1)
    assert starts[-1] == date(1986, 1, 1)                              # carve-out on the last curve date
    with pytest.raises(ConfigError):
        bk.window_starts(c, rets, FlatCurves(0.04, last=date(1980, 6, 1)), months)


def test_flat_history_reproduces_the_hand_calculation(cfg, schedule):
    """Zero stock returns, everything in Treasuries at a flat 4%: every window
    ends with exactly the contributions grown at 4% and the 4% reserve cost."""
    r = math.log(1.04)
    rets = returns_dict(date(1989, 1, 1), date(2000, 1, 1))
    c = with_changes(cfg, {"backtest.first_start": date(1990, 1, 1), "wind_down.risk_dial": 0.0})
    curves = FlatCurves(r)
    starts = bk.window_starts(c, rets, curves, 72)
    market = bk.build_market(c, schedule, starts, rets, curves)
    assert market.rates == pytest.approx(0.04, abs=1e-12)
    assert market.bills == pytest.approx(1.04 ** (1 / 12) - 1, abs=1e-12)
    res = bk.replay(c, schedule, starts, market).result
    assert res.total == pytest.approx(300_000 * 1.04 ** 6 + 150_000 * 1.04 ** 5, rel=1e-9)
    assert res.reserve == pytest.approx(RESERVE_AT_4, abs=1)
    assert res.success.all()


def test_market_uses_each_windows_own_months(cfg, schedule):
    rets = {k: (k[0] - 1980) / 1000 + k[1] / 100_000 for k in returns_dict(date(1979, 1, 1), date(2000, 1, 1))}
    c = with_changes(cfg, {"backtest.first_start": date(1990, 1, 1)})
    curves = FlatCurves(0.04)
    starts = bk.window_starts(c, rets, curves, 72)[:3]
    m = bk.build_market(c, schedule, starts, rets, curves)
    assert m.growth[1, 0] == rets[(1990, 2)] and m.growth[1, 13] == rets[(1991, 3)]
    assert m.prior[0, -1] == rets[(1989, 12)]


def test_quote_check_uses_nothing_after_the_quote_date(cfg, schedule):
    """Changing what happens AFTER the quote date must not change the quote."""
    c = with_changes(cfg, {"backtest.first_start": date(1990, 1, 1),
                           "backtest.calibration.paths_per_window": 200})
    rets = returns_dict(date(1989, 1, 1), date(2000, 1, 1), 0.005)
    curves = FlatCurves(math.log(1.045))
    starts = bk.window_starts(c, rets, curves, 72)[:4]
    k = month_starts(c["monte_carlo"]["start"], c["decision_dates"]["reserve_carve_out"]).index(
        c["monte_carlo"]["quote"]["date"])
    a = bk.build_market(c, schedule, starts, rets, curves)
    b = bk.build_market(c, schedule, starts, rets, curves)
    b.growth[:, k:] = -0.05                                           # a crash after the quote
    qa = bk.calibrate(c, schedule, bk.replay(c, schedule, starts, a))
    qb = bk.calibrate(c, schedule, bk.replay(c, schedule, starts, b))
    assert np.array_equal(qa.floor, qb.floor) and np.array_equal(qa.ceiling, qb.ceiling)
    assert (qa.floor <= qa.ceiling).all()
    assert not np.array_equal(qa.result.facility, qb.result.facility)  # ...but the outcome does change


def test_policy_menu_runs_every_policy_and_dial(cfg, schedule):
    c = with_changes(cfg, {"backtest.first_start": date(1990, 1, 1)})
    rets = returns_dict(date(1989, 1, 1), date(1998, 1, 1), 0.006)
    curves = FlatCurves(math.log(1.05))
    starts = bk.window_starts(c, rets, curves, 72)
    runs = bk.policy_menu(c, schedule, starts, bk.build_market(c, schedule, starts, rets, curves))
    assert len(runs) == len(c["monte_carlo"]["policies"]) * len(c["monte_carlo"]["risk_dials"])
