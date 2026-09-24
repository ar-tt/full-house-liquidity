"""Step 5: the Monte Carlo, on small made-up markets we control."""
import copy
import math
from datetime import date, timedelta

import numpy as np
import pytest

from fhl import montecarlo as mc
from fhl.config import ConfigError, with_changes
from fhl.curves import FlatCurve
from fhl.prices import Bar
from fhl.wind_down import status, targets

START, CARVE = date(2027, 1, 1), date(2033, 1, 1)
MONTHS = 72
RESERVE_AT_4 = 421_766.58


@pytest.fixture
def calm(cfg):
    """Settings for hand-checkable runs: no rate moves, 4% rates, loose risk controls."""
    c = copy.deepcopy(cfg)
    c["monte_carlo"]["rates"].update(vol_per_year=0.0, long_run=0.04)
    c["risk_controls"]["vol_target"] = [{"from": date(2020, 1, 1), "vol": 10.0}]
    c["risk_controls"]["turnover_cap"] = 100.0
    return c


def market(monthly_return, paths=3, months=MONTHS, vol=0.15):
    g = np.full((paths, months), monthly_return, dtype=float)
    return mc.Market(g, np.zeros((paths, months)), vol)


# --- building blocks ---------------------------------------------------------------

def test_fast_wind_down_matches_the_live_rule(cfg, schedule):
    rng = np.random.default_rng(1)
    for d in (date(2027, 1, 1), date(2028, 3, 1), date(2030, 6, 1), date(2031, 9, 1), date(2032, 12, 1)):
        for rt in (None, date(2028, 1, 1)):
            v, y = rng.uniform(1e5, 1.2e6, 50), rng.uniform(0.0, 0.09, 50)
            fast = targets(schedule, cfg, d, v, y, rt)
            slow = [status(a, 0, schedule, cfg, d, FlatCurve(b), rt).target for a, b in zip(v, y)]
            assert fast == pytest.approx(slow, abs=1e-9)


def test_month_starts():
    ds = mc.month_starts(date(2032, 11, 1), CARVE)
    assert ds == [date(2032, 11, 1), date(2032, 12, 1), CARVE]
    with pytest.raises(ConfigError):
        mc.month_starts(date(2027, 1, 15), CARVE)


def test_monthly_returns_include_dividends():
    d = date(2020, 1, 30)
    bars = [Bar(d, 100, 100, 100, 100, 1), Bar(d + timedelta(days=1), 100, 100, 100, 100, 1),
            Bar(date(2020, 2, 28), 99, 99, 99, 99, 1, dividend=2.0), Bar(date(2020, 3, 31), 110, 110, 110, 110, 1)]
    assert mc.monthly_returns(bars) == pytest.approx([0.01, 110 / 99 - 1])


def test_vol_target_steps_down(cfg):
    assert mc.vol_target_on(cfg, date(2029, 12, 1)) == 0.12
    assert mc.vol_target_on(cfg, date(2032, 1, 1)) == 0.08


def test_bootstrap_keeps_months_together_and_is_recentred(cfg):
    hist = np.expm1(np.linspace(-0.05, 0.08, 120))       # log returns rise by a fixed step each month
    m = mc.make_market(cfg, 36, 400, 7, hist)
    logs = np.log1p(m.growth)
    steps = np.diff(logs[:, :12], axis=1)                 # inside one 12-month block ...
    assert np.allclose(steps, steps[0, 0])                # ... months follow each other in history order
    target = math.log(1.07) / 12
    assert logs.mean() == pytest.approx(target, abs=0.002)


def test_same_seed_same_answer(cfg):
    a = mc.make_market(cfg, 24, 50, 5, np.linspace(-0.05, 0.08, 60))
    b = mc.make_market(cfg, 24, 50, 5, np.linspace(-0.05, 0.08, 60))
    assert np.array_equal(a.growth, b.growth) and np.array_equal(a.rate_shocks, b.rate_shocks)


def test_unknown_return_model_is_refused(cfg):
    with pytest.raises(ConfigError, match="bootstrap or lognormal"):
        mc.make_market(with_changes(cfg, {"monte_carlo.growth_returns.model": "vibes"}), 12, 5, 1)


# --- the simulation by hand -----------------------------------------------------------

def test_all_treasuries_by_hand(calm, schedule):
    """Dial 0: everything in the ladder and bills at a steady 4%. On 2033-01-01
    the portfolio is exactly the contributions grown at 4%, and the facility
    is that minus the 4% reserve anchor."""
    res = mc.simulate(with_changes(calm, {"wind_down.risk_dial": 0.0}), schedule, market(0.0), START, 300_000, 0.04)
    expected_total = 300_000 * 1.04 ** 6 + 150_000 * 1.04 ** 5
    assert res.total == pytest.approx(expected_total, rel=1e-9)
    assert res.reserve == pytest.approx(RESERVE_AT_4, abs=1)
    assert res.facility == pytest.approx(res.total - res.reserve, rel=1e-12)
    assert res.success.all() and (res.ladder_owned == 1).all()


def test_identical_markets_give_identical_paths(calm, schedule):
    res = mc.simulate(calm, schedule, market((1.07) ** (1 / 12) - 1), START, 300_000, 0.04)
    assert np.ptp(res.facility) < 1e-6 and res.success.all()


def test_facility_is_zero_when_the_reserve_is_not_covered(calm, schedule):
    crash = market(-0.03)                                   # stocks fall 3% every month for six years
    c = with_changes(calm, {"risk_controls.drawdown_trigger": 0.99, "wind_down.hedge_ramp": []})
    res = mc.simulate(c, schedule, crash, START, 300_000, 0.04)
    assert not res.funded.any() and (res.facility == 0).all() and not res.success.any()


def test_selling_stocks_in_a_slump_to_finish_the_ladder_fails_the_path(calm, schedule):
    g = np.full((2, MONTHS), 0.01)
    g[:, -3:] = -0.08                                       # a slump just before the carve-out
    slump = mc.Market(g, np.zeros((2, MONTHS)), 0.15)
    no_ramp = with_changes(calm, {"wind_down.hedge_ramp": [], "risk_controls.drawdown_trigger": 0.99})
    locked = with_changes(calm, {"wind_down.hedge_ramp": [{"from": date(2020, 1, 1), "share": 1.0}],
                                 "risk_controls.drawdown_trigger": 0.99})
    r1 = mc.simulate(no_ramp, schedule, slump, START, 300_000, 0.04)
    r2 = mc.simulate(locked, schedule, slump, START, 300_000, 0.04)
    assert r1.funded.all() and r1.forced.all() and not r1.success.any()
    assert (r2.ladder_owned == 1).all() and r2.success.all()


def test_drawdown_rule_caps_growth_at_half_until_recovery(calm, schedule):
    g = np.full((1, MONTHS), 0.0)
    g[0, 2] = -0.40                                         # a crash in March 2027
    res = mc.simulate(calm, schedule, mc.Market(g, np.zeros((1, MONTHS)), 0.15), START, 300_000, 0.04)
    assert res.growth_share[1, 0] > 0.5                      # before the crash
    assert (res.growth_share[3:10, 0] <= 0.5 + 1e-9).all()   # after it, and it never recovers here
    assert res.dd_capped[3:].min() == 1.0


def test_volatility_target_caps_growth(cfg, schedule):
    c = with_changes(cfg, {"monte_carlo.rates.vol_per_year": 0.0})
    res = mc.simulate(c, schedule, market(0.005, vol=0.24), START, 300_000, 0.04)
    assert res.growth_share[0] == pytest.approx(0.12 / 0.24)   # first year: 12% target / 24% volatility


def test_turnover_never_exceeds_the_cap_in_any_12_months(cfg, schedule):
    c = with_changes(cfg, {"monte_carlo.rates.vol_per_year": 0.0})
    rng = np.random.default_rng(3)
    wild = mc.Market(rng.normal(0.0, 0.09, (200, MONTHS)), np.zeros((200, MONTHS)), 0.15)
    res = mc.simulate(c, schedule, wild, START, 300_000, 0.04)
    rolling = np.array([res.turnover[max(0, k - 11):k + 1].sum(axis=0) for k in range(MONTHS)])
    assert rolling.max() <= c["risk_controls"]["turnover_cap"] + 1e-9


def test_values_are_recorded_on_request(calm, schedule):
    res = mc.simulate(with_changes(calm, {"wind_down.risk_dial": 0.0}), schedule, market(0.0), START,
                      300_000, 0.04, record=(date(2031, 1, 1),))
    v, y = res.recorded[date(2031, 1, 1)]
    assert v == pytest.approx(300_000 * 1.04 ** 4 + 150_000 * 1.04 ** 3, rel=1e-9)
    assert y == pytest.approx(0.04)


# --- the outputs ---------------------------------------------------------------------

def test_quote_is_the_20th_and_70th_percentile(calm, schedule):
    rng = np.random.default_rng(4)
    m = mc.Market(rng.normal(0.006, 0.04, (500, 24)), np.zeros((500, 24)), 0.15)
    q = mc.quote(calm, schedule, m, date(2031, 1, 1), 580_000, 0.04)
    assert q.floor == pytest.approx(np.percentile(q.result.facility, 20))
    assert q.ceiling == pytest.approx(np.percentile(q.result.facility, 70))
    assert q.floor < q.ceiling
    assert q.sentence().startswith("80% confident") and "50% chance" in q.sentence()


def test_menu_picks_the_most_facility_money_that_meets_the_threshold(cfg, schedule):
    rng = np.random.default_rng(5)
    c = with_changes(cfg, {"monte_carlo.risk_dials": [1.0, 0.0]})
    m = mc.Market(np.expm1(rng.normal(math.log(1.07) / 12, 0.05, (300, MONTHS))),
                  rng.standard_normal((300, MONTHS)), 0.17)
    runs = mc.policy_menu(c, schedule, m, START, 300_000, 0.045)
    assert len(runs) == len(c["monte_carlo"]["policies"]) * 2
    for th in (0.5, 0.9, 0.99):
        best = mc.best_for(runs, th)
        ok = [r for r in runs if r.success >= th]
        assert best.median == max(r.median for r in ok)
    assert mc.best_for(runs, 1.01) is None


def test_policies_in_config_are_valid(cfg):
    for p in cfg["monte_carlo"]["policies"]:
        with_changes(cfg, p["changes"])
    with pytest.raises(ConfigError, match="unknown setting"):
        with_changes(cfg, {"wind_down.hedge_rampp": []})


def test_live_settings_are_the_full_ramp_at_dial_075(cfg, schedule):
    """Simulating the config as it stands must equal the menu's full ramp at 0.75."""
    rng = np.random.default_rng(6)
    m = mc.Market(np.expm1(rng.normal(math.log(1.07) / 12, 0.045, (300, MONTHS))),
                  rng.standard_normal((300, MONTHS)), 0.155)
    live = mc.simulate(cfg, schedule, m, START, 300_000, 0.045)
    full = next(p for p in cfg["monte_carlo"]["policies"] if p["code"] == "F")
    menu = mc.simulate(mc.policy_config(cfg, full, 0.75), schedule, m, START, 300_000, 0.045)
    assert np.array_equal(live.facility, menu.facility) and np.array_equal(live.success, menu.success)
