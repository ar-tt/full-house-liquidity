"""Backtest 3 (Range Table rules) building blocks, on made-up returns."""
from datetime import date, timedelta

import numpy as np
import pytest

from fhl import backtest_range as br


def test_drawdown_by_hand():
    r = np.array([[0.10], [-0.50], [0.20]])          # 1 -> 1.1 -> 0.55 -> 0.66
    assert br.drawdown(r)[0] == pytest.approx(0.5)
    assert br.drawdown(np.array([[0.01], [0.02]]))[0] == 0


def test_years_text():
    assert br.years_text({2002, 2003, 2004, 2008, 2010, 2011}) == "2002-04, 2008, 2010-11"
    assert br.years_text(set()) == "never"


def test_month_starts_need_history_before_and_after():
    d0 = date(2020, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(400)]
    idx = br.month_starts_idx(dates, back=60, forward=100)
    assert all(i >= 60 and i + 100 < len(dates) for i in idx)
    assert [dates[i].day for i in idx] == [1] * len(idx)


def test_identical_assets_have_correlation_one_and_one_bet():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 300)
    R = np.column_stack([x, x, x])
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(300)]
    path = br.corr_path(dates, R, 60, 100)
    assert all(c == pytest.approx(1) and e == pytest.approx(1) for _, c, e in path)


def test_sector_cap_bins_cover_every_random_mix(cfg):
    rng = np.random.default_rng(2)
    R = rng.normal(0.0004, 0.01, (600, 9))
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(600)]
    res = br.sector_cap_test(dates, R, cfg)
    months = len(br.month_starts_idx(dates, 60, cfg["backtest"]["range"]["forward_days"]))
    assert sum(len(v) for v in res.values()) == months * cfg["backtest"]["range"]["random_portfolios"]
    assert all(v.shape[1] == 3 for v in res.values())
