"""Backtest 2 (Position Clock timing) on made-up price paths."""
from datetime import date, timedelta

import numpy as np
import pytest

from fhl import backtest_clock as bc
from fhl.prices import Bar


def path(closes, start=date(2000, 1, 3), spread=0.004, dividends=None):
    out, d = [], start
    for i, c in enumerate(closes):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append(Bar(d, c, c * (1 + spread), c * (1 - spread), c, 1e6, (dividends or {}).get(i, 0.0)))
        d += timedelta(days=1)
    return out


def test_total_return_index_adds_dividends():
    tr = bc.total_return_index(path([100, 100, 99], dividends={2: 1.0}))
    assert tr == pytest.approx([1.0, 1.0, 1.0])


def test_flat_market_all_three_ways_end_equal(cfg):
    """No price moves, no interest: every way in ends at exactly $1."""
    bars = path([100.0] * 700)
    e = bc.run(bars, np.zeros(len(bars)), cfg)
    for h in cfg["backtest"]["clock"]["horizons_days"]:
        assert e.lump[h] == pytest.approx(1.0) and e.thirds[h] == pytest.approx(1.0)
        assert e.clock[h] == pytest.approx(1.0)


def test_waiting_cash_earns_interest(cfg):
    bars = path([100.0] * 700)
    e = bc.run(bars, np.full(len(bars), 0.0001), cfg)
    h = min(cfg["backtest"]["clock"]["horizons_days"])
    assert (e.thirds[h] > e.lump[h]).all()               # flat prices: waiting in bills wins


def test_in_a_steady_rally_rsi_holds_the_clock_back_until_the_window_ends(cfg):
    bars = path([100 * 1.004 ** i for i in range(700)])  # RSI pinned at 100
    e = bc.run(bars, np.zeros(len(bars)), cfg)
    assert (e.rsi > 70).all()
    assert e.clock_first_delay.min() >= 38               # about 8 weeks of trading days
    h = max(cfg["backtest"]["clock"]["horizons_days"])
    assert (e.clock[h] < e.lump[h]).all()                # waiting in a rally costs money


def test_regimes_from_trend(cfg):
    icfg = cfg["position_clock"]["indicators"]
    up = bc.regimes(path([100 * 1.003 ** i for i in range(300)]), icfg)
    down = bc.regimes(path([100 * 0.997 ** i for i in range(300)]), icfg)
    assert up[-1] == "uptrend" and down[-1] == "downtrend" and up[50] == "no_trend"


def test_decision_uses_yesterdays_signals_not_todays(cfg):
    """If yesterday's RSI says 'buy today', a price spike TODAY (which would
    push today's RSI over 70) must not stop today's purchase."""
    base = [100.0 + np.sin(i / 7) for i in range(700)]
    a = bc.run(path(base), np.zeros(700), cfg)
    i = next(k for k in range(150, 400) if a.rsi[k] < 60 and a.clock_first_delay[k] == 0)
    day = a.dates[i]
    spiked = list(base)
    j = [b.date for b in path(base)].index(day)
    spiked[j] = 150.0
    b = bc.run(path(spiked), np.zeros(700), cfg)
    assert b.clock_first_delay[i] == 0                    # bought on the decision day anyway
    assert b.rsi[i + 1] > 70                              # ...even though today's RSI is now hot
