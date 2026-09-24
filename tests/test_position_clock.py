"""The Position Clock: tranche timing, RSI, sizing, the wind-down wait, and
the Range Table's wind-down ceiling. Indicator readings are set by hand here
(the indicator maths is tested in test_indicators.py)."""
import copy
from datetime import date, timedelta

import pytest

import fhl.pillars.position_clock as pc
from fhl.combine import HOLD, decide
from fhl.curves import FlatCurve
from fhl.indicators import Snapshot
from fhl.pillars.base import BLOCK, WAIT, Context
from fhl.pillars.position_clock import PositionClock
from fhl.pillars.range_table import RangeTable
from fhl.portfolio import Entry, Portfolio, Proposal
from fhl.prices import Bar, InMemoryPrices
from conftest import sec

TODAY = date(2027, 6, 30)
SECS = {"S1": sec("S1", "Industrials"), "S2": sec("S2", "Health Care"),
        "ETF1": sec("ETF1", "Diversified", type="etf"),
        "STRIP": sec("STRIP", "Government", type="bond", asset_class="bond", issuer="US Treasury", has_prices=False),
        "BONDS": sec("BONDS", "Government", type="etf", asset_class="bond")}
PRICES = InMemoryPrices({t: [Bar(TODAY - timedelta(days=1), 100, 101, 99, 100, 1e6),
                             Bar(TODAY, 100, 101, 99, 100, 1e6)] for t in SECS})


def snap(regime="no_trend", rsi=50.0, vol=0.25):
    return Snapshot(100.0, 95.0, 25.0, rsi, vol, regime, [])


@pytest.fixture
def clock(cfg, monkeypatch):
    def make(regime="no_trend", rsi=50.0, vol=0.25, today=TODAY):
        monkeypatch.setattr(pc, "snapshot", lambda bars, icfg: snap(regime, rsi, vol))
        return PositionClock({"name": "position", "weight": 25},
                             Context(cfg, SECS, PRICES, today, curve=FlatCurve(0.04)))
    return make


def pf(holdings=None, cash=1_000_000, entries=None):
    """A portfolio well under its growth target, so the wind-down doesn't interfere."""
    return Portfolio(holdings or {"BONDS": 500_000}, cash, entries or {}, date(2028, 1, 1))


def buy(t, amount, d=TODAY):
    return Proposal(t, "buy", amount, d)


def waits(res):
    return [r.code for r in res.fired if r.severity == WAIT]


# --- sizing ------------------------------------------------------------------------

def test_first_tranche_is_due_now_and_is_a_third(clock):
    c = clock(vol=0.25)
    p = pf()
    plan = c.plan("S1", p)
    assert plan.tranche_no == 1 and plan.due == TODAY
    assert plan.full_size == pytest.approx(0.035 * p.total)          # vol = reference vol
    assert plan.tranche_size == pytest.approx(plan.full_size / 3)
    assert waits(c.evaluate(buy("S1", plan.tranche_size), p)) == []


@pytest.mark.parametrize("vol, weight", [(0.50, 0.0175), (0.125, 0.05), (0.90, 0.01), (None, 0.035)])
def test_size_is_inverse_to_volatility_within_bounds(clock, vol, weight):
    p = pf()
    assert clock(vol=vol).plan("S1", p).full_size == pytest.approx(weight * p.total)


def test_etfs_use_their_own_sizing(clock):
    p = pf()
    assert clock(vol=0.16).plan("ETF1", p).full_size == pytest.approx(0.15 * p.total)


def test_existing_holding_counts_toward_the_full_size(clock):
    p = pf({"BONDS": 500_000, "S1": 30_000})
    plan = clock(vol=0.25).plan("S1", p)
    assert plan.full_size == pytest.approx(0.035 * p.total - 30_000)


def test_already_full_waits(clock):
    p = pf({"BONDS": 500_000, "S1": 60_000})
    assert waits(clock().evaluate(buy("S1", 1_000), p)) == ["full_size"]


def test_order_bigger_than_the_tranche_waits(clock):
    c, p = clock(), pf()
    t = c.plan("S1", p).tranche_size
    assert waits(c.evaluate(buy("S1", t * 1.09), p)) == []           # inside the 10% tolerance
    assert waits(c.evaluate(buy("S1", t * 1.5), p)) == ["tranche_size"]


# --- timing ------------------------------------------------------------------------

def entry(weeks_ago, done=1, full=30_000):
    return {"S1": Entry(TODAY - timedelta(weeks=weeks_ago), done, full)}


@pytest.mark.parametrize("regime, spacing", [("uptrend", 2), ("no_trend", 3), ("downtrend", 4)])
def test_regime_sets_the_spacing(clock, regime, spacing):
    c = clock(regime)
    early = c.evaluate(buy("S1", 10_000), pf(entries=entry(spacing - 1)))
    on_time = c.evaluate(buy("S1", 10_000), pf(entries=entry(spacing)))
    assert waits(early) == ["tranche_due"] and waits(on_time) == []


def test_rsi_under_40_pulls_the_tranche_forward_a_week(clock):
    p = pf(entries=entry(2))                                          # no trend: due at 3 weeks
    assert waits(clock("no_trend", rsi=50).evaluate(buy("S1", 10_000), p)) == ["tranche_due"]
    assert waits(clock("no_trend", rsi=35).evaluate(buy("S1", 10_000), p)) == []


def test_rsi_over_70_holds_even_the_first_tranche(clock):
    assert waits(clock(rsi=75).evaluate(buy("S1", 5_000), pf())) == ["rsi_delay"]


def test_after_eight_weeks_everything_is_due_whatever_rsi_says(clock):
    p = pf(entries=entry(8))
    assert waits(clock("downtrend", rsi=80).evaluate(buy("S1", 10_000), p)) == []


def test_last_tranche_is_whatever_is_left(clock):
    plan = clock().plan("S1", pf(entries=entry(6, done=2, full=30_000)))
    assert plan.tranche_no == 3 and plan.tranche_size == pytest.approx(10_000)


def test_a_finished_entry_starts_fresh(clock):
    plan = clock().plan("S1", pf(entries=entry(10, done=3)))
    assert plan.tranche_no == 1 and plan.new_entry


def test_timing_score_follows_regime_and_rsi(clock):
    assert clock("uptrend").plan("S1", pf()).score == 80
    assert clock("downtrend", rsi=30).plan("S1", pf()).score == 45 + 15
    assert clock("no_trend", rsi=75).plan("S1", pf()).score == 65 - 25


# --- what the clock leaves alone ----------------------------------------------------

def test_sells_are_not_timed(clock):
    res = clock(rsi=90).evaluate(Proposal("S1", "sell", 1_000, TODAY), pf({"S1": 5_000}))
    assert res.score is None and waits(res) == []


def test_treasuries_follow_the_ladder_not_the_clock(clock):
    res = clock(rsi=90).evaluate(buy("STRIP", 50_000), pf())
    assert res.score is None and waits(res) == []


# --- long hand inside the clock -----------------------------------------------------

def test_buy_waits_if_it_lifts_growth_above_the_wind_down_target(clock):
    c = clock()
    p = pf({"BONDS": 450_000, "S2": 500_000}, cash=50_000)           # 50% growth
    wd = c.wind_down(p)
    room = (wd.target + 0.02 - wd.growth_share) * p.total             # up to target + 2 points
    assert room > 0
    small, big = room - 5_000, room + 20_000
    assert "wind_down" not in waits(c.evaluate(buy("S1", small), p))
    assert "wind_down" in waits(c.evaluate(buy("S1", big), p))


def test_a_wait_holds_the_order_it_does_not_block(cfg, clock):
    c = clock(rsi=75)
    d = decide(buy("S1", 5_000), pf(), [c], cfg)
    assert d.status == HOLD and "RSI" in d.reason


# --- the Range Table's growth ceiling comes from the calendar -------------------------

def range_table(cfg, today, flag=True):
    c = copy.deepcopy(cfg)
    c["range_table"]["equity_max_from_wind_down"] = flag
    return RangeTable({"name": "range", "weight": 50, "structural": True},
                      Context(c, SECS, PRICES, today))


@pytest.mark.parametrize("today, ceiling", [
    (date(2029, 6, 1), 0.90), (date(2030, 6, 1), 0.80), (date(2032, 6, 1), 0.60), (date(2033, 2, 1), 0.0)])
def test_range_equity_ceiling_follows_the_calendar(cfg, today, ceiling):
    assert range_table(cfg, today).ac_budgets["equity"][1] == ceiling


def test_range_blocks_a_buy_over_the_calendar_ceiling(cfg):
    p = Portfolio({"S2": 750_000, "BONDS": 200_000}, 50_000)          # 75% growth
    rt = range_table(cfg, date(2030, 6, 1))
    res = rt.evaluate(buy("S1", 50_000, date(2030, 6, 1)), p)         # -> 80%: at the ceiling, fine
    assert "asset_class_budget" not in [r.code for r in res.fired if r.severity == BLOCK]
    p2 = Portfolio({"S2": 790_000, "BONDS": 160_000}, 50_000)
    res = rt.evaluate(buy("S1", 50_000, date(2030, 6, 1)), p2)        # -> 84%: over 80%
    assert "asset_class_budget" in [r.code for r in res.fired if r.severity == BLOCK]


def test_range_uses_the_fixed_budget_when_the_link_is_off(cfg):
    assert range_table(cfg, date(2032, 6, 1), flag=False).ac_budgets["equity"][1] == 0.90
