import copy
from pathlib import Path

import pytest

from fhl.cashflows import Schedule
from fhl.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture
def cfg(base_cfg):
    """A fresh copy each test, so a test can edit it freely."""
    return copy.deepcopy(base_cfg)


@pytest.fixture
def schedule(cfg):
    return Schedule.from_config(cfg)


@pytest.fixture
def carve(cfg):
    return cfg["decision_dates"]["reserve_carve_out"]


# ---------------------------------------------------------------------------
# Synthetic market data for the Range Table tests (no internet needed)
# ---------------------------------------------------------------------------
from datetime import date, timedelta  # noqa: E402

import numpy as np  # noqa: E402

from fhl.prices import Bar, InMemoryPrices  # noqa: E402
from fhl.securities import Security  # noqa: E402

START = date(2026, 1, 5)


def weekdays(n, start=START):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def bars_from_returns(rets, start=START, price=100.0):
    days = weekdays(len(rets) + 1, start)
    out = [Bar(days[0], price, price, price, price, 1e6)]
    for d, r in zip(days[1:], rets):
        price *= 1 + r
        out.append(Bar(d, price, price, price, price, 1e6))
    return out


def factor_prices(groups: dict, n_days=120, seed=7, vol=0.01):
    """groups: {group_name: (rho, [tickers])}. Tickers in a group share one
    market factor with pairwise correlation about rho; groups are independent."""
    rng = np.random.default_rng(seed)
    data = {}
    for rho, tickers in groups.values():
        f = rng.standard_normal(n_days)
        for t in tickers:
            e = rng.standard_normal(n_days)
            data[t] = bars_from_returns(vol * (np.sqrt(rho) * f + np.sqrt(1 - rho) * e))
    return InMemoryPrices(data)


def sec(ticker, sector="Industrials", region="US", type="stock", asset_class="equity",
        issuer=None, has_prices=True, sector_weights=None, region_weights=None, cik=None):
    return Security(ticker, ticker, type, asset_class, issuer or ticker,
                    sector_weights or {sector: 1.0}, region_weights or {region: 1.0}, has_prices, cik)
