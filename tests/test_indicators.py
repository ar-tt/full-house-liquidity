"""Price indicators, checked by hand and on made-up price paths."""
import math
from datetime import date, timedelta

import pytest

from fhl.indicators import adx, realized_vol, rsi, sma, snapshot
from fhl.prices import Bar

ICFG = {"trend_ma_days": 200, "adx_days": 14, "adx_trending_above": 20, "rsi_days": 14, "vol_days": 60}


def bars(closes, spread=0.5):
    d0 = date(2020, 1, 1)
    return [Bar(d0 + timedelta(days=i), c, c + spread, c - spread, c, 1e6) for i, c in enumerate(closes)]


def test_sma():
    assert sma([1, 2, 3, 4], 2) == 3.5
    assert sma([1, 2], 3) is None


def test_rsi_by_hand():
    # 2-day RSI of 1,2,1,2: first averages 0.5/0.5, then Wilder smoothing
    # gain (0.5+1)/2 = 0.75, loss (0.5+0)/2 = 0.25 -> RSI = 100 - 100/(1+3) = 75
    assert rsi([1, 2, 1, 2], 2) == pytest.approx(75)


def test_rsi_extremes():
    assert rsi(list(range(1, 40)), 14) == 100
    assert rsi(list(range(40, 1, -1)), 14) == pytest.approx(0)
    assert rsi([5.0] * 30, 14) == 50
    assert rsi([1, 2, 3], 14) is None


def test_adx_strong_steady_trend_is_high():
    assert adx(bars([100 + i for i in range(60)]), 14) == pytest.approx(100)


def test_adx_flat_market_is_zero():
    assert adx(bars([100.0] * 60, spread=0.0), 14) == 0.0


def test_adx_choppy_market_is_low():
    chop = [100 + (1 if i % 2 else -1) for i in range(80)]
    assert adx(bars(chop), 14) < 20


def test_adx_needs_enough_bars():
    assert adx(bars([100 + i for i in range(20)]), 14) is None


def test_realized_vol():
    closes = [100.0]
    for i in range(70):
        closes.append(closes[-1] * (1.01 if i % 2 else 1 / 1.01))
    v = realized_vol(bars(closes), 60)
    n = 60
    # log returns of exactly +-ln(1.01), average zero; sample standard deviation, annualized
    expected = math.log(1.01) * math.sqrt(n / (n - 1)) * math.sqrt(252)
    assert v == pytest.approx(expected, rel=1e-3)
    assert realized_vol(bars(closes[:30]), 60) is None


def test_regimes():
    up = snapshot(bars([100 * 1.002 ** i for i in range(260)]), ICFG)
    down = snapshot(bars([100 * 0.998 ** i for i in range(260)]), ICFG)
    flat = snapshot(bars([100 + (1 if i % 2 else -1) for i in range(260)]), ICFG)
    assert (up.regime, down.regime, flat.regime) == ("uptrend", "downtrend", "no_trend")


def test_short_history_is_treated_as_no_trend_and_says_so():
    s = snapshot(bars([100 + i for i in range(100)]), ICFG)
    assert s.regime == "no_trend" and s.ma is None and "under 200 days" in s.notes[0]


# --- one-pass series equal the single-day functions ------------------------------

def _wobbly(n=320, seed=9):
    import numpy as np
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    d0 = date(2020, 1, 1)
    return [Bar(d0 + timedelta(days=i), c, c * (1 + abs(e)), c * (1 - abs(e)), c, 1e6)
            for i, (c, e) in enumerate(zip(closes, rng.normal(0, 0.006, n)))]


def test_series_match_the_single_day_functions():
    from fhl.indicators import adx_series, rsi_series, sma_series
    b = _wobbly()
    closes = [x.close for x in b]
    sma_s, rsi_s, adx_s = sma_series(closes, 200), rsi_series(closes, 14), adx_series(b, 14)
    for i in (5, 14, 15, 28, 29, 30, 60, 199, 200, 250, len(b) - 1):
        sub = b[:i + 1]
        for got, want in ((sma_s[i], sma([x.close for x in sub], 200)),
                          (rsi_s[i], rsi([x.close for x in sub], 14)),
                          (adx_s[i], adx(sub, 14))):
            if want is None:
                assert math.isnan(got)
            else:
                assert got == pytest.approx(want, abs=1e-9)
