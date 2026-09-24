"""Backtest 4 (Convergence) building blocks, without the internet."""
from datetime import date, timedelta

import numpy as np
import pytest

from fhl import backtest_convergence as bcv
from fhl.prices import Bar, InMemoryPrices


def test_name_key_matches_reorganized_companies():
    assert bcv.name_key("WALT DISNEY CO/") == bcv.name_key("Walt Disney Co")
    assert bcv.name_key("Exxon Mobil Corporation") == bcv.name_key("ExxonMobil Holdings Corp")
    assert bcv.name_key("CIGNA CORP") == bcv.name_key("Cigna Group")
    assert bcv.name_key("Raytheon Co") != bcv.name_key("RTX Corp")


def test_sector_for_uses_the_first_matching_range(cfg):
    rows = cfg["backtest"]["convergence"]["sic_sectors"]
    assert bcv.sector_for(2834, rows) == "Health Care"          # pharmaceuticals
    assert bcv.sector_for(6324, rows) == "Health Care"          # health insurers, before Financials
    assert bcv.sector_for(6021, rows) == "Financials"           # banks
    assert bcv.sector_for(6798, rows) == "Real Estate"          # REITs
    assert bcv.sector_for(7372, rows) == "Information Technology"
    assert bcv.sector_for(None, rows) is None


def bars(start, n, daily=0.0005, div_on=None):
    out, p = [], 100.0
    for i in range(n):
        d = start + timedelta(days=i)
        p *= 1 + daily
        out.append(Bar(d, p, p, p, p, 1e6, 1.0 if d == div_on else 0.0))
    return out


def test_forward_return_needs_the_whole_year():
    d0 = date(2020, 4, 1)
    prices = InMemoryPrices({"FULL": bars(date(2020, 1, 1), 600), "SHORT": bars(date(2020, 1, 1), 200)})
    full = bcv.forward_return(prices, "FULL", d0, 365)
    assert full == pytest.approx(1.0005 ** 365 - 1, rel=1e-6)
    assert bcv.forward_return(prices, "SHORT", d0, 365) is None       # stopped trading mid-year
    assert bcv.forward_return(prices, "NONE", d0, 365) is None


def test_group_returns_when_scores_predict_perfectly():
    rng = np.random.default_rng(1)
    rows = []
    for y in range(2015, 2020):
        for i in range(30):
            s = rng.normal()
            rows.append({"date": date(y, 4, 1), "score": s, "fwd": 0.1 * s})
    g = bcv.group_returns(rows, "score", 3)
    assert (g["groups"][0] > g["groups"][1]).all() and (g["groups"][1] > g["groups"][2]).all()
    assert g["ic"] == pytest.approx(1.0)


def test_group_returns_skip_missing_values():
    rows = [{"date": date(2020, 4, 1), "score": None if i % 2 else i, "fwd": 0.01 * i} for i in range(40)]
    g = bcv.group_returns(rows, "score", 3)
    assert len(g["ic"]) == 1
